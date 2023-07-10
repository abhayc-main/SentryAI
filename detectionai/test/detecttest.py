import cv2
from mtcnn import MTCNN

"""
detector = MTCNN()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB color for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    results = detector.detect_faces(frame_rgb)

    if results:
        # Extract the bounding box coordinates of the first detected face
        x, y, w, h = results[0]['box']

        # Add 200 to width and height and make the face at the center
        x_large = max(0, x - 100)
        y_large = max(0, y - 100)
        w_large = min(frame.shape[1], w + 200)
        h_large = min(frame.shape[0], h + 200)

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x_large, y_large), (x_large + w_large, y_large + h_large), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
"""

import cv2

# Load the Haar Cascade XML file for face detection
cascade_path = './haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a bounding box around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
