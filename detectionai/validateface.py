import os
import cv2
from deepface import DeepFace

def create_user_folder(user_name):
    # Create a folder with the user's name
    user_folder = f"./data/{user_name}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    # Initialize counter for labeled images
    image_count = 1
    
    # Iterate over the images
    for i in range(1, 6):
        image_path = f"./data/images/image{i}.jpg"  # Change the path to your image directory
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Extract the face from the image
        detected_face = DeepFace.extract_face(img, detector_backend='mtcnn')
        
        # If face is detected, save the original image
        if detected_face is not None:
            save_path = f"{user_folder}/{user_name}{image_count}.jpg"
            cv2.imwrite(save_path, img)
            image_count += 1

    # Check if at least 5 images with faces were found
    if image_count > 1:
        print(f"Folder '{user_name}' created with {image_count-1} labeled images.")
    else:
        print("No face found in the images.")

# Example usage
user_name = input("Enter user name: ")
create_user_folder(user_name)
