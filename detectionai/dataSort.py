import os
import shutil

def create_folders(images_path):
    # Iterate over the images in the specified directory
    for image_file in os.listdir(images_path):
        if image_file.endswith(".jpg"):
            # Extract the name from the image file
            name = os.path.splitext(image_file)[0]
            # Create a new folder based on the name
            new_folder = os.path.join(images_path, name)
            os.makedirs(new_folder, exist_ok=True)
            # Move the image file to the new folder
            shutil.move(os.path.join(images_path, image_file), os.path.join(new_folder, image_file))

if __name__ == "__main__":
    # Specify the path to the directory containing the images
    images_path = "detectionai/data"
    create_folders(images_path)

