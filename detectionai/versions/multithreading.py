# Version 2 implementation

import cv2
import numpy as np
import face_recognition
from multiprocessing import Manager
import os

# Define the dataset directory
dataset_dir = './data/people'

# Define the dataset directory
label_dir = './data/people'

known_faces = []
known_labels = []
dataset_processed = False


while True:
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        print("Processing image:", img_path)  # Debugging statement
        try:
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_faces.append(face_encoding)
                known_labels.append(label_name)
            else:
                print("Error: No face found in", img_path)
                os.remove(img_path)
        except Exception as e:
            print("Error processing image:", img_path)
            print("Error message:", str(e))
            os.remove(img_path)    
    break
    

# Initialize shared variable for inter-process communication
manager = Manager()
read_frame_list = manager.list([None])

def capture():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        read_frame_list[0] = frame

def Process():
    while True:
        frame = read_frame_list[0]
        if frame is not None:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_labels[best_match_index]
                    verification_status = "Verified"
                else:
                    verification_status = "Not Verified"

                top, right, bottom, left = face_locations[best_match_index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name} ({verification_status})", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # Start the capture process
    capture_process = Process(target=capture)
    capture_process.start()

    # Start the processing process
    process_process = Process(target=Process)
    process_process.start()

    capture_process.join()
    process_process.join()

    cv2.destroyAllWindows()
