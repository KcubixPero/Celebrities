import os
import cv2 as cv
import numpy as np
      
people = []
DIR = "Faces\\train"
for i in os.listdir(DIR):
    people.append(i)   
print(people)

features = []       #Stores cropped face regions
labels = []         #Stores numeric labels (0,1,2,...) corresponding to people   
haar_cascade = cv.CascadeClassifier(r'Celebrities/haar_face.xml')    #Loads a Haar Cascade classifier XML file (haar_face.xml), 
                                                                     #used for detecting faces in images.

def create_train():
    for person in people:
        path = os.path.join(DIR, person)   # full folder path of the person
        label = people.index(person)       # numeric label for that person
        
        for img in os.listdir(path):            # loop over each image in that folder
            img_path = os.path.join(path, img)  # full folder path of the image
            
            img_array = cv.imread(img_path)                   # read the image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)  # convert to grayscale
            
            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4
            )  # detect faces in the image
            
            for (x,y,w,h) in faces_rect:               # for each detected face
                faces_roi = gray[y:y+h, x:x+w]         # crop face (region of interest)
                features.append(faces_roi)             # add cropped face to features
                labels.append(label)                   # add corresponding label
 
create_train()  
print('Training done -------------------------------')     
          
features = np.array(features, dtype = 'object') 
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Now, we will train the recognizer of our data(features and labels)

face_recognizer.train(features, labels)

face_recognizer.save(r'Celebrities/face_trained.yml')    #.yml is often used to store trained model data 
np.save(r'Celebrities/features.npy', features)
np.save(r'Celebrities/labels.npy', labels)
