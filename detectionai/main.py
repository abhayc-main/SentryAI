import face_recognition
import cv2
import numpy as np

# Set up paths
dataset_dir = './data/dataset/'  # Directory containing the training dataset

# Load the dataset of known faces
known_faces = []
known_labels = []
for label_name in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label_name)
    if os.path.isdir(label_dir):
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)
            known_labels.append(label_name)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Resize frame to improve performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Initialize an empty list to store the names of verified faces
    verified_names = []
    
    for face_encoding in face_encodings:
        # Compare the face encoding with the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unauthorized"
        
        if True in matches:
            # Face recognized as one of the known faces
            match_index = matches.index(True)
            name = known_labels[match_index]
            verified_names.append(name)
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, verified_names):
        # Scale back up the face locations since the frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + ' Verified', (left + 6, bottom - 6), font, 0.5, (0, 255, 0), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
