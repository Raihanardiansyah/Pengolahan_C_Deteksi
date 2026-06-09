import cv2
import pytesseract
import matplotlib.pyplot as plt

# Jalur ke tesseract yang dapat dieksekusi
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_plate_number(image_path):
    # Muat gambar
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Konversi ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Terapkan Gaussian Blur untuk menghilangkan noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi (Canny) untuk menyorot kontur pelat
    edges = cv2.Canny(blurred, 100, 200)

    # Temukan kontur untuk menemukan plat nomor
    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Urutkan kontur berdasarkan area (urutan menurun)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate_contour = None
    for contour in contours:
        # Perkiraan kontur ke poligon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Periksa apakah kontur memiliki 4 simpul
        # (yang mungkin persegi panjang, khas untuk pelat)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        # Gambar kotak pembatas di sekitar plat nomor yang terdeteksi
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]

        # Terapkan ambang batas untuk membinarisasi area pelat
        _, thresh = cv2.threshold(
            plate_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Lakukan OCR pada area pelat yang terdeteksi
        # Perlakukan sebagai satu kata
        plate_number = pytesseract.image_to_string(thresh, config='--psm 8')
        return plate_number.strip()
    else:
        return "License plate not detected"


# Berikan jalur gambar
image_path = "plat.jpg"  # Ganti dengan jalur gambar Anda

# Deteksi dan cetak nomor plat
plate_number = detect_plate_number(image_path)
print("Detected Plate Number:", plate_number)