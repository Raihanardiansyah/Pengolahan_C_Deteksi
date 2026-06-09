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