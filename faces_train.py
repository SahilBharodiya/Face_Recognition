import os
import cv2 as cv
import numpy as np

people = ['Abdul Kalam', 'Bill Gates', 'Elon Musk', 'Mukesh Ambani', 'Ratan Tata', 'Satya Nadella', 'Sundar Pichai']

DIR = r'E:\\Programming\\AI-ML\\openCV\\models\\Person Recognition\\train'

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

fetures = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                fetures.append(faces_roi)
                labels.append(label)


create_train()
print("training Done")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

fetures = np.array(fetures, dtype='object')
labels = np.array(labels)

face_recognizer.train(fetures, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', fetures)
np.save('labels.npy', labels)
