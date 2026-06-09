# Pengolahan_C_Deteksi

## Deteksi Pada Pejalan Kaki

Kode

```Python
import cv2
import imutils

# Menginisialisasi HOG Person Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar
image = cv2.imread('walking.jpg')

# Mengubah ukuran gambar
image = imutils.resize(image, width=min(400, image.shape[1]))

# Mendeteksi pejalan kaki
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(4, 4),
    scale=1.05
)

# Menggambar kotak deteksi
for (x, y, w, h) in regions:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

# Menampilkan hasil
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Kode ini digunakan untuk mendeteksi keberadaan pejalan kaki pada sebuah gambar menggunakan metode Histogram of Oriented Gradients (HOG) yang dikombinasikan dengan Support Vector Machine (SVM) bawaan OpenCV. Program diawali dengan mengimpor library cv2 dan imutils, kemudian menginisialisasi objek HOG dan memuat model deteksi manusia yang telah dilatih sebelumnya. Setelah itu, gambar dibaca dari file walking.png dan ukurannya diperkecil agar proses deteksi menjadi lebih cepat dan efisien. Fungsi detectMultiScale() digunakan untuk mencari area-area pada gambar yang diduga mengandung manusia. Jika objek manusia ditemukan, program akan menggambar kotak berwarna merah di sekeliling objek tersebut menggunakan fungsi cv2.rectangle(). Terakhir, hasil deteksi ditampilkan dalam sebuah jendela dan pengguna dapat melihat hasilnya hingga tombol keyboard ditekan. Program ini menunjukkan bagaimana teknik visi komputer dapat digunakan untuk mengenali manusia secara otomatis pada citra digital.

## Deteksi Pada Video Orang Berjalan

```Python
import cv2
import imutils

# Menginisialisasi HOG Person Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membuka video
cap = cv2.VideoCapture('walking.mp4')

while cap.isOpened():

    # Membaca frame video
    ret, image = cap.read()

    if ret:

        image = imutils.resize(
            image,
            width=min(400, image.shape[1])
        )

        # Mendeteksi pejalan kaki
        (regions, _) = hog.detectMultiScale(
            image,
            winStride=(4, 4),
            padding=(4, 4),
            scale=1.05
        )

        # Menggambar kotak deteksi
        for (x, y, w, h) in regions:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

        # Menampilkan hasil
        cv2.imshow("Image", image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
```

Kode ini menerapkan konsep yang sama seperti Contoh 1, namun digunakan pada sebuah video sehingga deteksi dapat dilakukan secara berkelanjutan pada setiap frame video. Program diawali dengan menginisialisasi HOG Descriptor dan model SVM bawaan OpenCV untuk mendeteksi manusia. Selanjutnya, video dibuka menggunakan cv2.VideoCapture() dan setiap frame dibaca secara berulang menggunakan perulangan while. Setiap frame yang berhasil dibaca akan diubah ukurannya agar proses komputasi lebih ringan, kemudian fungsi detectMultiScale() digunakan untuk mendeteksi pejalan kaki pada frame tersebut. Jika ditemukan objek manusia, program akan menggambar kotak berwarna merah pada posisi objek yang terdeteksi. Hasil deteksi kemudian ditampilkan secara real-time sehingga pengguna dapat melihat proses pelacakan manusia selama video diputar. Program akan terus berjalan hingga video selesai atau pengguna menekan tombol Q untuk menghentikan proses. Implementasi ini sering digunakan dalam sistem pengawasan, analisis lalu lintas, dan teknologi kendaraan otonom yang memerlukan deteksi manusia secara langsung dari aliran video.

## Deteksi Plat Nomor Kendaraan

kode 
```python
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
```

Program ini digunakan untuk mendeteksi dan membaca nomor plat kendaraan dari sebuah gambar menggunakan library OpenCV dan Tesseract OCR. Proses dimulai dengan membaca gambar kendaraan, kemudian mengubahnya menjadi citra grayscale untuk mempermudah pengolahan. Selanjutnya, dilakukan pengurangan noise menggunakan Gaussian Blur dan deteksi tepi menggunakan metode Canny Edge Detection untuk menemukan bentuk-bentuk objek pada gambar. Setelah itu, program mencari kontur yang menyerupai bentuk persegi panjang sebagai kandidat area plat nomor. Jika area plat berhasil ditemukan, bagian tersebut dipotong dan diproses menggunakan teknik thresholding agar karakter pada plat terlihat lebih jelas. Hasil citra kemudian dikirim ke Tesseract OCR untuk mengenali huruf dan angka yang terdapat pada plat nomor. Nomor plat yang berhasil dibaca akan ditampilkan sebagai output, sedangkan jika tidak ditemukan area yang sesuai maka program akan menampilkan pesan bahwa plat nomor tidak terdeteksi. Program ini merupakan penerapan pengolahan citra digital dan Optical Character Recognition (OCR) yang banyak digunakan pada sistem parkir otomatis, pengawasan lalu lintas, dan identifikasi kendaraan.
