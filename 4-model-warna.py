import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr_to_cmyk(img):
    bgr = img.astype(float) / 255.0
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    
    K = 1 - np.max(bgr, axis=2)
    
    denom = 1 - K
    denom[denom == 0] = 1e-5
    
    C = (1 - R - K) / denom
    M = (1 - G - K) / denom
    Y = (1 - B - K) / denom
    
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    return CMYK

def bgr_to_hsi(img):
    bgr = img.astype(float) / 255.0
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    
    I = (R + G + B) / 3.0
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    sum_rgb = R + G + B
    sum_rgb[sum_rgb == 0] = 1e-5  # Cegah pembagian nol
    S = 1 - (3 / sum_rgb) * min_rgb
    
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-5
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    
    H = np.zeros_like(I)
    H[B > G] = 2 * np.pi - theta[B > G]
    H[B <= G] = theta[B <= G]
    
    H_deg = (H / (2 * np.pi)) * 255
    S_scaled = S * 255
    I_scaled = I * 255
    
    HSI = np.dstack((H_deg, S_scaled, I_scaled)).astype(np.uint8)
    return HSI

def main():
    img_path = 'foto.jpg'
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Gambar '{img_path}' tidak ditemukan.")
        return

    width, height = 1280, 720
    img_bgr = cv2.resize(img_bgr, (width, height))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_cmyk = bgr_to_cmyk(img_bgr)
    img_hsi = bgr_to_hsi(img_bgr)

    plt.figure(figsize=(10, 10))
    
    plt.subplot(231)
    plt.imshow(img_rgb)
    plt.title('RGB')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(img_bgr)
    plt.title('BGR')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(img_hsv, cmap='gray')
    plt.title('HSV')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(img_cmyk, cmap='gray')
    plt.title('CMYK')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(img_hsi, cmap='gray')
    plt.title('HSI')
    plt.axis('off')
    
    plt.show()
    

if __name__ == "__main__":
    main()