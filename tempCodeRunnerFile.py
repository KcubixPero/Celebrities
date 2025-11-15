import numpy as np
import cv2 as cv

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_cascade = cv.CascadeClassifier(r'Celebrities\haar_face.xml')

# Load the trained face recognizer model

face_recognizer = face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'Celebrities\face_trained.yml')

img = cv.imread(r'C:\Users\Asus\Desktop\OpenCV\Faces\val\madonna\4.jpg')  
if img is None:
    print("Error: Image not found. Check the path again.")
    exit()

cv.imshow('Face', img)   
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]   # Region of Interest (detected face)

    # Predict the person
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence = {confidence}')

    # Draw rectangle and put label text
    
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.putText(img, str(people[label]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
