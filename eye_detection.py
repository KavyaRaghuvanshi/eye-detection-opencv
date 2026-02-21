import cv2

# Load eye detection model
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, frame = camera.read()
    if not success:
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Eye Detection", frame)

    # Press 'q' to close
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()