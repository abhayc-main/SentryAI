import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("./firebase/serviceAccount.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "sentryai-777b5.appspot.com"
})

# Importing student images
folderPath = 'data'
subfolders = [f.path for f in os.scandir(folderPath) if f.is_dir()]
imgList = []
imgPaths = []
studentIds = []
for subfolder in subfolders:
    label = os.path.basename(subfolder)
    imgFiles = os.listdir(subfolder)
    for imgFile in imgFiles:
        imgPath = os.path.join(subfolder, imgFile)
        imgList.append(cv2.imread(imgPath))
        imgPaths.append(imgPath)
        studentIds.append(label)


def findEncodings(imagesList):
    encodeList = []
    for img, imgPath in zip(imagesList, imgPaths):
        try:
            if img is None:
                print(f"Empty image at path: {imgPath}. Skipping...")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faceEncodings = face_recognition.face_encodings(img)
            if len(faceEncodings) > 0:
                encode = faceEncodings[0]
                encodeList.append(encode)
            else:
                print(f"No face found in image: {imgPath}. Skipping...")
        except Exception as e:
            print(f"Error processing image: {imgPath}\n{str(e)}")
            continue
    return encodeList



print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
