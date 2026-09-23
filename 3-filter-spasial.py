import cv2
import numpy as np
import matplotlib.pyplot as plt

# Pada Foto
img = cv2.imread('foto.jpg')
if img is None:
    print("No picture found.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean_filter = cv2.blur(gray, (5, 5))

gaussian_filter = cv2.GaussianBlur(gray, (5, 5), 0)

median_filter = cv2.medianBlur(gray, 5)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(
    np.float32(np.abs(sobel_x)), np.float32(np.abs(sobel_y))
)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

kernel_sharpen = np.array([
    [0, -1, 0], 
    [-1, 5, -1], 
    [0, -1, 0]
])
sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

plt.figure(figsize=(10, 10))

plt.subplot(241)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(242)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(243)
plt.imshow(mean_filter, cmap='gray')
plt.title('Mean Filter')
plt.axis('off')

plt.subplot(244)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(245)
plt.imshow(median_filter, cmap='gray')
plt.title('Median Filter')
plt.axis('off')

plt.subplot(246)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Filter')
plt.axis('off')

plt.subplot(247)
plt.imshow(laplacian, cmap='gray') 
plt.title('Laplacian Filter')
plt.axis('off')

plt.subplot(248)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Pada Live Video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to exit the video stream.")
print("Press '1' for Mean Filter")
print("Press '2' for Gaussian Filter")
print("Press '3' for Median Filter")
print("Press '4' for Sobel Filter")
print("Press '5' for Laplacian Filter")
print("Press '6' for Sharpened Image")
print("Press '7' for Gray Scale Video.")
print("Press '0' for Original Video.")

filter_type = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame.")
        break

    normal_video = frame.copy()
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if filter_type == 1:
        frame = cv2.blur(gray_video, (5, 5)) # cv2.blur(src, (width, height)) / Harus angka positif, semakin besar semakin kuat blurnya
        filter_name = 'Mean Filter'
    elif filter_type == 2:
        frame = cv2.GaussianBlur(gray_video, (5, 5), 0) # cv2.GaussianBlur(src, (width, height), sigmaX) / Harus angka positif dan ganjil, semakin besar semakin kuat blurnya  
        filter_name = 'Gaussian Filter'
    elif filter_type == 3:
        frame = cv2.medianBlur(gray_video, 5) # cv2.medianBlur(src, ksize) / Harus angka positif, ganjil, dan lebih besar dari 1
        filter_name = 'Median Filter'
    elif filter_type == 4:
        sobel_x_video = cv2.Sobel(gray_video, cv2.CV_64F, 1, 0, ksize=5) # cv2.Sobel(src, ddepth, dx, dy, ksize) / Harus angka positif dan ganjil (standar ksize = 3, kalau -1 namanya jadi Scharr Filter)
        sobel_y_video = cv2.Sobel(gray_video, cv2.CV_64F, 0, 1, ksize=5)
        frame = cv2.magnitude(
            np.float32(np.abs(sobel_x_video)), np.float32(np.abs(sobel_y_video))
        )
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filter_name = 'Sobel Filter'
    elif filter_type == 5:
        frame = cv2.Laplacian(gray_video, cv2.CV_64F) # cv2.Laplacian(src, ddepth)
        frame = cv2.convertScaleAbs(frame)
        filter_name = 'Laplacian Filter'
    elif filter_type == 6:
        kernel_sharpen_video = np.array([
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
        ])
        frame = cv2.filter2D(gray_video, -1, kernel_sharpen_video) # cv2.filter2D(src, ddepth, kernel) / ddepth = -1 untuk mempertahankan tipe data asli
        filter_name = 'Sharpened Image'
    elif filter_type == 7:
        frame = gray_video
        filter_name = 'Gray Scale Video'
    else:
        frame = normal_video
        filter_name = 'Original Video'

    colored_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.putText(colored_frame, filter_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Filter Spasial untuk Video', colored_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        filter_type = 1
    elif key == ord('2'):
        filter_type = 2
    elif key == ord('3'):
        filter_type = 3
    elif key == ord('4'):
        filter_type = 4
    elif key == ord('5'):
        filter_type = 5
    elif key == ord('6'):
        filter_type = 6
    elif key == ord('7'):
        filter_type = 7
    elif key == ord('0'):
        filter_type = 0

cap.release()
cv2.destroyAllWindows()