import cv2
import numpy as np

im = cv2.imread('logo_its.png', cv2.IMREAD_GRAYSCALE)
r = np.arange(256, dtype=np.float64)

lut_negatif = (255 - r)

def rapikan(lut):
    return np.clip(np.floor(lut + 0.5), 0, 255).astype(np.uint8)

hasil_negatif = cv2.LUT(im, rapikan(lut_negatif))

cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)