import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load img
img = cv2.imread('your_path_file', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.show()

# Binarization
_, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

# Plot binary img
plt.imshow(binary, cmap='gray')
plt.title("Binary image")
plt.show()

# Find Countours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# Adjust a circunference
(x, y), radius = cv2.minEnclosingCircle(contour)
circle_center = (int(x), int(y))
radius = int(radius)

cv2.circle(img,circle_center,radius,(0,255,0),2)
height, width = img.shape
frame_redimensionado = cv2.resize(img, (round(height/5), round(width/5)))
cv2.imshow("Image",frame_redimensionado)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Contact point with surface: Approximately y + r
bottom_points = contour[np.where(contour[:, :, 1] >= np.max(contour[:, :, 1]) - 2)]

# Fitts a line between the points
[vx, vy, x0, y0] = cv2.fitLine(bottom_points, cv2.DIST_L2, 0, 0.01, 0.01)
slope = vy.item() / vx.item()

# Tangent with horizontal
angle_rad = np.arctan(slope)
angle_deg = np.degrees(angle_rad)

# Considers the contact angle as |90 - |θ||
contact_angle = abs(90 - abs(angle_deg))

print(f'Approximate contact angle: {contact_angle:.2f} º')
