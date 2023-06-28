import cv2
import os
from deepface import DeepFace

# Define the path to the data folder
data_folder = "./data/"

# Load images from the data folder into a database
database = {}
for user_folder in os.listdir(data_folder):
    user_path = os.path.join(data_folder, user_folder)
    if os.path.isdir(user_path):
        images = []
        for image_file in os.listdir(user_path):
            image_path = os.path.join(user_path, image_file)
            if os.path.isfile(image_path):
                images.append(image_path)
        database[user_folder] = images

# Start the webcam
cap = cv2.VideoCapture(0)

# Face detection and recognition loop
while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Perform face detection using a face detection algorithm (e.g., RetinaFace or MTCNN)
    faces = DeepFace.extract_faces(frame, detector_backend='retinaface')

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Save the face image as a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_face_file:
            temp_face_path = temp_face_file.name
            cv2.imwrite(temp_face_path, face)
        # Apply face recognition by comparing the face embeddings with the database
        recognized = False
        for user, images in database.items():
            result = DeepFace.verify(img1_path = face, img2_path=images)
            if result["verified"]:
                # Display a message indicating authorized access
                cv2.putText(frame, f"Authorized: {user}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                recognized = True
                break
        
        if not recognized:
            # Display a message indicating unauthorized access
            cv2.putText(frame, "Unauthorized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()

cv2.destroyAllWindows()
