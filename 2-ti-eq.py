import math
import cv2
import numpy as np

im = cv2.imread('logo_its.png', cv2.IMREAD_GRAYSCALE)
r = np.arange(256, dtype=np.float64)

lut_negatif = (255 - r)

# Catatan di Kelas
# def rapikan(lut):
#     return np.clip(np.floor(lut + 0.5), 0, 255).astype(np.uint8)

# hasil_negatif = cv2.LUT(im, rapikan(lut_negatif))

# Transformasi Negatif
def negatif(im):
    tinggi, lebar = im.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)
    for i in range(tinggi):
        for j in range(lebar):
            hasil[i, j] = 255 - im[i, j]
    return hasil

# Transformasi Logaritmik
def logaritmik(im):
    tinggi, lebar = im.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)
    c = 255 / math.log(256)
    for i in range(tinggi):
        for j in range(lebar):
            pixel = int(c * math.log(1 + int(im[i, j])))

            if pixel > 255:
                pixel = 255

            hasil[i, j] = pixel
    return hasil

# Transformasi Gamma
def gamma(im, nilai_gamma):
    tinggi, lebar = im.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            r = im[i, j] / 255.0
            pixel = int(255 * (r ** nilai_gamma))

            if pixel > 255:
                pixel = 255

            hasil[i, j] = pixel
    return hasil

# Thresholding
def thresholding(im, threshold):
    tinggi, lebar = im.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            if im[i, j] >= threshold:
                hasil[i, j] = 255
            else:
                hasil[i, j] = 0
    return hasil

#Ekualisasi Histogram
def ekualisasi(im):
    tinggi, lebar = im.shape
    jumlah_pixel = tinggi * lebar

    hist = [0] * 256

    for i in range(tinggi):
        for j in range(lebar):
            hist[im[i, j]] += 1

    cdf = [0] * 256
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    lut = [0] * 256
    for i in range(256):
        lut[i] = int((cdf[i] / jumlah_pixel) * 255)
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            hasil[i, j] = lut[im[i, j]]
    return hasil

hasil_negatif = negatif(im)
hasil_logaritmik = logaritmik(im)
hasil_gamma = gamma(im, 2.2)
hasil_thresholding = thresholding(im, 128)
hasil_ekualisasi = ekualisasi(im)

cv2.waitKey(0)
cv2.destroyAllWindows()