import cv2 as cv
import numpy as np
import os

people = ["Quang", "NotQuang"]
DIR = r'./images/train'

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []

def train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            if img == '.DS_Store':
                continue

            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if person == "Quang":
                for (x,y,w,h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)
            elif person == "NotQuang":
                for i in range(len(faces_rect)):
                    (x,y,w,h) = faces_rect[i]
                    faces_roi = gray[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)


train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

print('---------------Training done ---------------')

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)