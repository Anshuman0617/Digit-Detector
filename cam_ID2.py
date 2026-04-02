import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# EXACT MODEL ARCHITECTURE
# -------------------------

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN,self).__init__()

        self.convolution1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )
        
        self.convolution2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        self.fullconn1 = nn.Linear(
            in_features=64*5*5,
            out_features=128
        )
        
        self.fullconn2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self,x):
        x = self.pool(F.relu(self.convolution1(x)))
        x = self.pool(F.relu(self.convolution2(x)))
        x = torch.flatten(input=x, start_dim=1)
        x = F.relu(self.fullconn1(x))
        x = self.fullconn2(x)
        return x


# -------------------------
# LOAD MODEL
# -------------------------

model = DigitCNN()
model.load_state_dict(
    torch.load("best_model.pth", map_location=torch.device("cpu"))
)
model.eval()


# -------------------------
# CAPTURE IMAGE
# -------------------------

cap = cv2.VideoCapture(0)
captured = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera - Press C", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        captured = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

if captured is None:
    print("No image captured.")
    exit()


# -------------------------
# PREPROCESSING
# -------------------------

gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive threshold to handle uneven lighting and shadows
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    15,
    5
)

kernel = np.ones((3, 3), np.uint8)

# Thicken the pen strokes to match MNIST and connect broken parts
# (MORPH_OPEN was destroying the very thin pen lines)
thresh = cv2.dilate(thresh, kernel, iterations=1)


# -------------------------
# FIND CONTOURS
# -------------------------

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


# -------------------------
# PROCESS CONTOURS
# -------------------------

detected_digits = []

for cnt in contours:

    area = cv2.contourArea(cnt)

    x, y, w, h = cv2.boundingRect(cnt)

    # Ignore contours that touch the very edges of the camera frame
    # (These are almost always background objects, fingers, or paper edges)
    img_h, img_w = thresh.shape
    if x <= 5 or y <= 5 or (x + w) >= (img_w - 5) or (y + h) >= (img_h - 5):
        continue

    # Use bounding box size instead of just area, since thin lines have small area
    if h < 20 or w < 5 or h > 150: # lowered max height to filter out big background lines
        continue
        
    # Still restrict extreme contour areas (but lower the min threshold)
    if area < 20 or area > 10000:
        continue

    aspect_ratio = w / float(h)

    # Relaxed aspect ratio to allow thin 1s and wider handwritten digits
    if aspect_ratio < 0.05 or aspect_ratio > 1.5:
        continue

    # Solidity filter
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        continue

    solidity = area / hull_area

    # Relaxed solidity threshold (thin digits like '2' or '3' have very low solidity)
    if solidity < 0.15:
        continue

    # Draw box
    cv2.rectangle(
        captured,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # -------------------------
    # Extract Digit With Padding
    # -------------------------

    pad = 10

    y1 = max(y - pad, 0)
    y2 = min(y + h + pad, thresh.shape[0])
    x1 = max(x - pad, 0)
    x2 = min(x + w + pad, thresh.shape[1])

    digit = thresh[y1:y2, x1:x2]

    if digit.size == 0:
        continue

    # -------------------------
    # Resize to 28x28
    # -------------------------

    digit_resized = cv2.resize(digit, (28, 28))

    # -------------------------
    # Normalize (match training)
    # -------------------------

    digit_model = digit_resized.astype(np.float32) / 255.0

    # ONLY enable this if you used MNIST normalization in training:
    # digit_model = (digit_model - 0.1307) / 0.3081

    # NEW FILTER: Pixel Density
    # Digits are made of strokes, so they only occupy a certain percentage of the box.
    # Solid blocks (like fingers) or empty noise will be thrown out here.
    pixel_density = np.mean(digit_model)
    if pixel_density < 0.05 or pixel_density > 0.45:
        continue

    # -------------------------
    # Convert to tensor
    # -------------------------

    digit_tensor = torch.from_numpy(digit_model).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(digit_tensor)
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted_digit_tensor = torch.max(probabilities, dim=1)
        predicted_digit = predicted_digit_tensor.item()
        confidence = max_prob.item()

    # Reject predictions where the model is not highly confident
    if confidence < 0.8:
        continue

    # Create a label with both the digit and the confidence percentage
    label = f"{predicted_digit} ({int(confidence * 100)}%)"

    cv2.putText(
        captured,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    detected_digits.append({'digit': predicted_digit, 'x': x, 'y': y, 'h': h})


# -------------------------
# PRINT IN ORDER
# -------------------------

if detected_digits:
    # 1. Sort all digits top-to-bottom
    detected_digits.sort(key=lambda d: d['y'])

    lines = []
    current_line = [detected_digits[0]]

    # 2. Group digits into lines based on vertical position
    for d in detected_digits[1:]:
        # If the new digit's Y is close to the current line's Y (within 70% of its height)
        # it belongs to the same line
        if abs(d['y'] - current_line[0]['y']) < (d['h'] * 0.70):
            current_line.append(d)
        else:
            lines.append(current_line)
            current_line = [d]
    lines.append(current_line)

    print("\n--- Detected Numbers ---")
    for line in lines:
        # 3. Sort digits in the same line from left-to-right
        line.sort(key=lambda d: d['x'])
        line_str = " ".join(str(d['digit']) for d in line)
        print(line_str)
else:
    print("\nNo numbers detected.")


# -------------------------
# DISPLAY RESULTS
# -------------------------

cv2.imshow("Threshold", thresh)
cv2.imshow("Final Detection", captured)

cv2.waitKey(0)
cv2.destroyAllWindows()
