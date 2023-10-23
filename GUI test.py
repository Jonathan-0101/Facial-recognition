import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
import os

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()


def recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(1)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if (confidence < 100 and confidence > 10 or confidence == 0):
                id = cursor.execute("SELECT userName FROM People WHERE userId = ?", (id,)).fetchone()[0]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(
                img,
                str(id),
                (x+5, y-5),
                font,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                img,
                str(confidence),
                (x+5, y+h-5),
                font,
                1,
                (255, 255, 0),
                1
            )

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def addUserLive(window):
    cursor.execute("SELECT * FROM People")
    face_id = len(cursor.fetchall()) + 1
    cam = cv2.VideoCapture(1)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    # Create a window to enter the name
    nameWindow = tk.Toplevel(window)
    # create a label for the window
    label = tk.Label(nameWindow, text="Enter the name")
    nameEntryBox = tk.Entry(nameWindow)
    # create a button to submit the name
    nameEntryButton = tk.Button(nameWindow, text="Submit", command=lambda: nameWindow.destroy(), width=20, height=2, padx=10, pady=10)
    # pack the label, entry box and button
    label.pack()
    nameEntryBox.pack()
    nameEntryButton.pack()
    # Set the name to the value entered in the entry box
    face_name = nameEntryBox.get()
    nameWindow.mainloop()
    # Insert the new user into the database
    cursor.execute("INSERT INTO People (userId, userName) VALUES (?, ?)", (face_id, face_name))
    conn.commit()
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


def addImages():
    print("Add images")


# Make a GUI window
window = tk.Tk()
window.title("Menu")
window.geometry("200x300")


spacer = tk.Label(window, text="", height=1)

button1 = tk.Button(window, text="Live Recognition", command=recognition, width=20, height=2, padx=10, pady=10)

button2 = tk.Button(window, text="Add Person", command=lambda: addUserLive(window), width=20, height=2, padx=10, pady=10)

button3 = tk.Button(window, text="Add Photos Of Exising Person", command=addImages, width=20, height=2, padx=10, pady=10)

button4 = tk.Button(window, text="Exit", command=window.destroy, width=20, height=2, padx=10, pady=10)

# Place the buttons in the window
spacer.pack()
button1.pack()
button2.pack()
button3.pack()
button4.pack()


window.mainloop()
conn.close()
