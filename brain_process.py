import pydicom
import matplotlib.pyplot as plt

#1
ds = pydicom.dcmread("E:/Anonymized_20250522/series-00000/image-00111.dcm")
image = ds.pixel_array

plt.imshow(image, cmap='gray')
plt.title('Original X-Ray')
plt.axis("off")
plt.show(block=True)

#2
import cv2
import numpy as np

# нормалізація інтенсивності пікселів
normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
normalized = np.uint8(normalized)
plt.imshow(normalized, cmap='gray')
plt.title("Normalized X-Ray")
plt.axis("off")
plt.show(block=True)

#3
blurred = cv2.GaussianBlur(image, (5, 5), 0)
plt.imshow(blurred, cmap='gray')
plt.title("Denoised X-Ray")
plt.axis("off")
plt.show(block=True)

#4
#усереднення фону + віднімання (корекція віньєтування)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
corrected = clahe.apply(blurred)
plt.imshow(corrected, cmap='gray')
plt.title("Contrast Enhaced X-Ray")
plt.axis("off")
plt.show(block=True)