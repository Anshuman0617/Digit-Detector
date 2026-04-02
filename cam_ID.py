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

# Proper Otsu threshold
_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((3, 3), np.uint8)

# Remove small noise
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Reconnect strokes
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


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

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 300 or area > 8000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    if h == 0:
        continue

    aspect_ratio = w / h

    if aspect_ratio < 0.2 or aspect_ratio > 1.0:
        continue

    # Solidity filter
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        continue

    solidity = area / hull_area

    if solidity < 0.4:
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

    # -------------------------
    # Convert to tensor
    # -------------------------

    digit_tensor = torch.from_numpy(digit_model).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(digit_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()

    cv2.putText(
        captured,
        str(predicted_digit),
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


# -------------------------
# DISPLAY RESULTS
# -------------------------

cv2.imshow("Threshold", thresh)
cv2.imshow("Final Detection", captured)

cv2.waitKey(0)
cv2.destroyAllWindows()