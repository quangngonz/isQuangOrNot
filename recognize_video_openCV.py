import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = ["Quang", "NotQuang"]

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        label, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(img, str(people[label]), (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    # Display
    cv.imshow('img', img)
    # Stop if escape key is pressed
    k = cv.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
