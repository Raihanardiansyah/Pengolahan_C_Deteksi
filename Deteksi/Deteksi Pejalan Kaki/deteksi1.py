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