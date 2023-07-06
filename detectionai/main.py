import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy as np
import os
import dlib

# Define the dataset directory
dataset_dir = './data/people'

known_faces = []
known_labels = []
for label_name in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label_name)
    if os.path.isdir(label_dir):
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


# Initialize shared variables for inter-process communication
manager = Manager()
read_frame_list = manager.list([None] * cpu_count())  # List to store frames for reading
write_frame_list = manager.list([None] * cpu_count())  # List to store frames for writing
frame_delay = 0.01  # Delay between frames for smoother video
is_exit = False  # Flag to indicate if the program should exit
buff_num = cpu_count()  # Buffer index for reading frames
read_num = cpu_count()  # Frame index for reading frames
write_num = 1  # Frame index for writing frames

# Set up the capture process
def capture(read_frame_list, is_exit, buff_num):
    video_capture = cv2.VideoCapture(0)
    while not is_exit.value:
        if buff_num.value != read_num.value:
            ret, frame = video_capture.read()
            read_frame_list[buff_num.value] = frame
            buff_num.value = next_id(buff_num.value, cpu_count())
        else:
            time.sleep(0.01)
    video_capture.release()

# Set up the processing processes
def process(worker_id, read_frame_list, write_frame_list, is_exit, read_num, buff_num, write_num):
    while not is_exit.value:
        while read_num.value != worker_id or read_num.value != prev_id(buff_num.value, cpu_count()):
            if is_exit.value:
                break
            time.sleep(0.01)

        time.sleep(frame_delay)
        frame_process = read_frame_list[worker_id]
        read_num.value = next_id(read_num.value, cpu_count())

        rgb_frame = frame_process[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_labels[best_match_index]
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        while write_num != worker_id:
            time.sleep(0.01)

        write_frame_list[worker_id] = frame_process
        write_num = next_id(write_num, cpu_count())

def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1

def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1

if __name__ == '__main__':
    set_start_method('spawn')

    # Start the capture process
    capture_process = Process(target=capture, args=(read_frame_list, is_exit, buff_num))
    capture_process.start()

    # Start the processing processes
    process_list = []
    for i in range(1, cpu_count() + 1):
        process_list.append(Process(target=process, args=(i, read_frame_list, write_frame_list, is_exit, read_num, buff_num, write_num)))
        process_list[-1].start()

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        frame = write_frame_list[cpu_count()]
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_exit = True
            break

    video_capture.release()

    # Wait for the processes to finish
    capture_process.join()
    for process in process_list:
        process.join()

    cv2.destroyAllWindows()

