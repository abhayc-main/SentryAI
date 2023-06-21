import cv2
import numpy as np
import argparse
import os
from deepface import DeepFace


def detect_and_match_faces(image, model):
  """Detects faces in an image and matches them with images in a dataset.

  Args:
    image: The image to detect faces in.
    model: The DeepFace model to use for detection and matching.

  Returns:
    A list of the identities of the people in the image.
  """

  faces = model.detect_faces(image)
  identities = []
  for face in faces:
    embedding = model.extract_face_embedding(image, face)
    identity = model.predict(embedding)
    identities.append(identity)

  return identities


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", required=True,
                      help="The image to detect faces in.")
  parser.add_argument("--model", required=True,
                      help="The DeepFace model to use for detection and matching.")
  args = parser.parse_args()

  image = cv2.imread(args.image)
  model = DeepFace.loadModel(args.model)

  identities = detect_and_match_faces(image, model)
  print("The identities of the people in the image are:", identities)


if __name__ == "__main__":
  main()


# Testing the accuracy

def test_accuracy(model):
  """Tests the accuracy of the model prediction.

  Args:
    model: The DeepFace model to use for testing.

  Returns:
    The accuracy of the model prediction.
  """

  correct = 0
  total = 0
  for filename in os.listdir("data"):
    if filename.endswith(".jpg"):
      image = cv2.imread(os.path.join("data", filename))
      faces = model.detect_faces(image)
      identity = model.predict(faces[0])
      if identity == filename.split(".")[0]:
        correct += 1
      total += 1

  accuracy = correct / total
  print("The accuracy of the model prediction is:", accuracy)
  