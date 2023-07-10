import os
import pyheif
from PIL import Image


def convert_heic_to_jpg(heic_path, parent_dir_name):
    # Open the HEIC image using pyheif
    heif_image = pyheif.read(heic_path)

    # Convert the HEIC image to PIL Image object
    image = Image.frombytes(
        heif_image.mode,
        heif_image.size,
        heif_image.data,
        "raw",
        heif_image.mode,
        heif_image.stride,
    )

    # Get the JPG filename with an incremental counter
    counter = 1
    while True:
        jpg_filename = f"{parent_dir_name}_{counter}.jpg"
        jpg_path = os.path.join(os.path.dirname(heic_path), jpg_filename)
        if not os.path.exists(jpg_path):
            break
        counter += 1

    # Save the PIL Image object as JPG, replacing the original HEIC file
    image.save(jpg_path, "JPEG")

    print(f"Converted: {heic_path} -> {jpg_path}")


def convert_directory(heic_directory):
    # Recursively convert HEIC images to JPG in the directory
    for root, dirs, files in os.walk(heic_directory):
        parent_dir_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(".heic"):
                heic_path = os.path.join(root, file)
                convert_heic_to_jpg(heic_path, parent_dir_name)


heic_directory = "./data"

convert_directory(heic_directory)
