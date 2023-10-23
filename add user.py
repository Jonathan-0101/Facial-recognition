import cv2
import os
import sqlite3
import numpy as np
from PIL import Image

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM People")
face_id = len(cursor.fetchall()) + 1
cam = cv2.VideoCapture(1)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_name = input('\n Enter name and press <return> ==>  ')
# Insert the new user into the database
cursor.execute("INSERT INTO People (userId, userName) VALUES (?, ?)", (face_id, face_name))
conn.commit()
conn.close()
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("Dataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# Path for face image database
path = 'Dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('Trainer/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
