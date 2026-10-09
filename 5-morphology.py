import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img_color = cv2.imread('foto.jpg')
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if img is None:
        print("Gambar tidak ditemukan!")
        return

    img = cv2.resize(img, (1280, 720))

    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary_img, kernel, iterations=1)
    dilation = cv2.dilate(binary_img, kernel, iterations=1)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(10, 10))

    plt.subplot(231)
    plt.imshow(img_color)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(erosion, cmap='gray')
    plt.title('Erosi')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilasi')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()