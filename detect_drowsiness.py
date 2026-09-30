import cv2
import pygame
import time

# -----------------------------
# ALARM
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load(r"D:\DRIVER_DROWSINESS_DETECTION\alarm.wav")

# -----------------------------
# HAAR CASCADES
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)
eye_closed_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    eyes_detected_total = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        eyes_detected_total += len(eyes)

    # -------------------------
    # GLOBAL EYE LOGIC
    # -------------------------

    if eyes_detected_total >= 1:
        eye_closed_start = None
        pygame.mixer.music.stop()

        cv2.putText(frame, "Eyes Open", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        if eye_closed_start is None:
            eye_closed_start = time.time()

        closed_time = time.time() - eye_closed_start

        if closed_time > 2:
            cv2.putText(frame, "DROWSINESS ALERT!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    cv2.imshow("Driver_Drowsiness_Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()