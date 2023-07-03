import cv2
from mtcnn import MTCNN
import os
from deepface import DeepFace
import time

# Set up video capture from the webcam
cap = cv2.VideoCapture(0)

# Create a directory to save input images
INPUT_DIR = './data/input/'
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

verification_images = {}
verification_dir = './data/people/'
for folder_name in os.listdir(verification_dir):
    folder_path = os.path.join(verification_dir, folder_name)
    if os.path.isdir(folder_path):
        verification_images[folder_name] = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.JPG'):
                file_path = os.path.join(folder_path, filename)
                verification_images[folder_name].append(file_path)


# Create an instance of the MTCNN face detector
detector = MTCNN()

start_time = time.time()

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

        # Save the cropped face region as the input image
        input_image = frame[y_large:y_large + h_large, x_large:x_large + w_large]
        cv2.imwrite(os.path.join(INPUT_DIR, 'input_image.jpg'), input_image)

        # Perform verification against all verification images
        # Perform verification against all verification images
    verified = False
    for person, image_paths in verification_images.items():
        for verification_image in image_paths:
            result = DeepFace.verify(img1_path=os.path.join(INPUT_DIR, 'input_image.jpg'), img2_path=verification_image)
            if result["verified"]:
                verified = True
                break

    
    if verified:
        print(f"Verified as {person}")
        break

    if not verified:
        print("Unauthorized Person")

    

    # Display the frame with face detections
    cv2.imshow('Real-time Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


end_time = time.time()

# Calculate the time taken for the algorithm to work
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
