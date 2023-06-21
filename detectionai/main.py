import cv2
import numpy as np

# Load the face detection algorithm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the model
model = cv2.face.LBPHFaceRecognizer_create()

# Load the known faces
known_faces = []
known_names = []
for name in ['John Doe', 'Jane Doe', 'Mary Smith']:
    image = cv2.imread('images/' + name + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        known_faces.append(face)
        known_names.append(name)

# Load the uploaded dataset
uploaded_faces = []
uploaded_names = []
for name in ['uploaded_face_1', 'uploaded_face_2', 'uploaded_face_3']:
    image = cv2.imread('images/' + name + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        uploaded_faces.append(face)
        uploaded_names.append(name)

# Train the model
model.train(known_faces, known_names)

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Recognize the faces in the frame
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        name = model.predict(face)

        # Check if the face is in the uploaded dataset
        if name in uploaded_names:
            name = 'Unknown (uploaded)'

        # Draw a rectangle around the face and label it with the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the video stream
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
