
from deepface import DeepFace


result = DeepFace.verify(img1_path="./data/input/input_image.jpg", img2_path="./data/people/3.JPG")

print(result)