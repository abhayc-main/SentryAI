import os
import shutil

"""
Make sure the image names are written with the format - "names.jpg" so it can look like this...

user
├── database
│   ├── Alice
│   │   ├── Alice1.jpg
│   │   ├── Alice2.jpg
│   ├── Bob
│   │   ├── Bob.jpg

"""

def create_folders(images_path):
    # Iterate over the images in the specified directory
    for image_file in os.listdir(images_path):
        if image_file.endswith(".jpg" or ".png" or ".PNG"):
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

