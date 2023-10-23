import os
import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog


conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"

def message(window, messageText):
    messageBox = tk.Toplevel(window)
    messageBox.title("Alert")
    messageBox.geometry("300x150")
    Label(messageBox, text=messageText, font=largefont, padx=10, pady=10).pack()
    Button(messageBox, text="OK", command=lambda: [messageBox.destroy()], pady=10, padx=10, height=1, width=20).pack()
    messageBox.mainloop()

def trainer(): # pottentially add in some error handling if images do not load
    path = 'Dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        return faceSamples, ids
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('Trainer/trainer.yml')

def recognizeFace(cam, faceCascade, minW, minH, recognizer, font):  
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

        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100 and confidence > 10):
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
    # Display the resulting frame in the tkinter window
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk

def liveRecognition(window): # todo
    print("Live Recognition")
    for widget in window.winfo_children():
        widget.destroy()
    window.geometry("1080x720")
    Label(window, text="Live Recogniton", font=largefont).pack()
    ImageLable = Label(window)
    ImageLable.pack()
    Button(window, text="Return To Menu", command=lambda: [menu(window)], pady=10, padx=10, height=1, width=20).pack()
    # Create a frame in the middle of the screen
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(1)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    imgtk = recognizeFace(cam, faceCascade, minW, minH, recognizer, font)
    ImageLable.imgtk = imgtk
    ImageLable.configure(image=imgtk)
    ImageLable.after(1, lambda: [recognizeFace(cam, faceCascade, minW, minH, recognizer, font)])
        
def photoRecognition(window): # todo
    print("Photo Recognition")

def addUserLive(window): # todo
    print("Add User Live")            
    
def uploadExistingImages(tree, personList, window):
    row = tree.focus()
    if len(row) == 0:
        return
    rowId = int(row.strip("I"))
    rowId = rowId - 1
    face_id = personList[rowId][0]
    imageFiles = filedialog.askopenfilenames()
    count = 0
    uploads = 0
    for filename in os.listdir("Dataset"):
        if filename.startswith("User."+str(face_id)+"."):
            count += 1
    for file in imageFiles:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite("Dataset/User." + str(face_id) + '.' +
                        str(count) + ".jpg", gray[y:y+h, x:x+w])
            uploads += 1
    messageText = str(uploads) + " images uploaded for \n" + str(personList[rowId][1])
    message(window, messageText)
    trainer()

def addImages(window):
    for widget in window.winfo_children():
        widget.destroy()
    window.geometry("480x735")
    Label(window, text="Select person", font=largefont, padx=10, pady=10).pack()
    frame = Frame(window)
    frame.pack(pady=20)
    tree = ttk.Treeview(frame, columns=(1, 2), show='headings', height=25)
    tree.pack(side=LEFT)
    sb = Scrollbar(frame, orient=VERTICAL)
    sb.pack(side=RIGHT, fill=Y)
    tree.config(yscrollcommand=sb.set)
    sb.config(command=tree.yview)
    tree["columns"] = ("1", "2")
    tree['show'] = 'headings'
    tree.heading("1", text="ID")
    tree.heading("2", text="Name")
    people = cursor.execute("SELECT * FROM People")
    people = cursor.fetchall()
    personList = []
    loopNum = 0
    for item in people:
        personId = item[0]
        name = item[1]
        personInfo = [personId, name]
        personList.append(personInfo)
        loopNum += 1
        pos = ("L", (loopNum))
        tree.insert("", "end", text=pos, values=([personId, name]))

    style = ttk.Style(window)
    style.theme_use("default")
    style.map("Treeview")
    Button(window, text="Upload Images", command=lambda: [uploadExistingImages(tree, personList, window)], font=largefont, pady=10, padx=10, height=1, width=20).pack()
    Label(window, text=" ").pack()
    Button(window, text="Return To Menu", command=lambda: [menu(window)], font=largefont, pady=10, padx=10, height=1, width=20).pack()
    window.mainloop()

def addPerson(window, newUserName):
    userId = cursor.execute("SELECT * FROM People ORDER BY userId DESC").fetchall()[0][0] +1
    newUserName = newUserName.get()
    cursor.execute("INSERT INTO People (userId, userName) VALUES (?, ?)", (userId, newUserName))
    conn.commit()
    imageFiles = filedialog.askopenfilenames()
    count = 0
    uploads = 0
    for file in imageFiles:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite("Dataset/User." + str(userId) + '.' +
                        str(count) + ".jpg", gray[y:y+h, x:x+w])
            uploads += 1
    messageText = str(uploads) + " images uploaded for \n" + str(newUserName)
    for widget in window.winfo_children():
        widget.destroy()
    Label(window, text=messageText, font=largefont, padx=10, pady=10).pack()
    Button(window, text="Add another person", command=lambda: [addUserFromImages(window)], font=largefont, pady=10, padx=10, height=1, width=20).pack()
    Label(window, text="").pack()
    Button(window, text="Return To Menu", command=lambda: [menu(window)], font=largefont, pady=10, padx=10, height=1, width=20).pack()
    trainer()
    
def addUserFromImages(window):
    for widget in window.winfo_children():
        widget.destroy()
    window.geometry("300x225")
    newUserName = StringVar()
    Label(window, text="Enter Name", font=largefont, padx=10, pady=10).pack()
    Entry(window, textvariable=newUserName, width=20, font=largefont).pack()
    Label(window, text=" ").pack()
    Button(window, text="Add Person", command=lambda: [addPerson(window, newUserName), newUserName.set("")], width=15, font=largefont, pady=10, padx=10, height=1).pack()
    Label(window, text=" ").pack()
    Button(window, text="Return To Menu", command=lambda: [menu(window)], width=15, font=largefont, padx=10, pady=10).pack()
    
def addUser(window):
    for widget in window.winfo_children():
        widget.destroy()
    addUserLiveButton = Button(window, text="Add Person Live", command=lambda: [addUserLive(window)], width=20, font=largefont, padx=10, pady=10)
    addUserFromImagesButton = Button(window, text="Add Person From Images", command=lambda: [addUserFromImages(window)], width=20, font=largefont, padx=10, pady=10)
    returnButton = Button(window, text="Return To Menu", command=lambda: [menu(window)], width=20, font=largefont, padx=10, pady=10)
    addUserLiveButton.place(relx=0.5, rely=0.3, anchor="center")
    addUserFromImagesButton.place(relx=0.5, rely=0.45, anchor="center")
    returnButton.place(relx=0.5, rely=0.9, anchor="center")

def menu(window):
    window.title("Menu")
    window.geometry("300x450")

    for widget in window.winfo_children():
        widget.destroy()

    button1 = Button(window, text="Live Recognition", command=lambda: [liveRecognition(window)], width=20, font=largefont, padx=10, pady=10)

    button2 = Button(window, text="Photo/Video Recogniton", command=lambda: [photoRecognition(window)], width=20, font=largefont, padx=10, pady=10)

    button3 = Button(window, text="Add Person", command=lambda: [addUser(window)], width=20, font=largefont, padx=10, pady=10)

    button4 = Button(window, text="Add Photos", command=lambda: [addImages(window)], width=20, font=largefont, padx=10, pady=10)

    button5 = Button(window, text="Exit", command=exit, width=20, font=largefont, padx=10, pady=10)

    button1.place(relx=0.5, rely=0.15, anchor="center")
    button2.place(relx=0.5, rely=0.3, anchor="center")
    button3.place(relx=0.5, rely=0.45, anchor="center")
    button4.place(relx=0.5, rely=0.6, anchor="center")
    button5.place(relx=0.5, rely=0.9, anchor="center")

window = tk.Tk()
largefont = ("Verdana", 12)
menu(window)
window.mainloop()
conn.close()
