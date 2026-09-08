import cv2
import matplotlib.pyplot as plt

def display_image(image_path):
    img = cv2.imread(image_path)

    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def filter_color_image(image_path, color):
    img = cv2.imread(image_path)
    [h,w,c] = img.shape

    if color == 'red':
        for i in range(h):
            for j in range (w):
                img[i, j, 1] = 0
                img[i, j, 0] = 0
    if color == 'green':
        for i in range(h):
            for j in range (w):
                img[i, j, 2] = 0
                img[i, j, 0] = 0
    if color == 'blue':
        for i in range(h):
            for j in range (w):
                img[i, j, 2] = 0
                img[i, j, 1] = 0
    else:
        print("Color not recognized. Please choose 'red', 'green', or 'blue'.")

    cv2.imshow(f"Filtered Image - {color}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_color_video(color):
    cap = cv2.VideoCapture(0)

    [h,w,c] = cap.read()[1].shape

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if color == 'red':
            for i in range(h):
                for j in range (w):
                    frame[i, j, 1] = 0
                    frame[i, j, 0] = 0
        elif color == 'green':
            for i in range(h):
                for j in range (w):
                    frame[i, j, 2] = 0
                    frame[i, j, 0] = 0
        elif color == 'blue':
            for i in range(h):
                for j in range (w):
                    frame[i, j, 2] = 0
                    frame[i, j, 1] = 0
        else:
            print("Color not recognized. Please choose 'red', 'green', or 'blue'.")

        cv2.imshow(f"Filtered Video - {color}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

display_image('logo_its.png')
filter_color_image('logo_its.png', 'red')
filter_color_video('blue')