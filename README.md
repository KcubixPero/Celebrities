Face Recognition Model Trainer (OpenCV + LBPH)

This project trains a face recognition model using OpenCV's LBPH (Local Binary Patterns Histogram) algorithm.
It detects faces using a Haar Cascade classifier, extracts face regions, labels them, and trains a recognizer model.

🚀 Features

Automatically loads training images from Faces/train/<person_name>/

Detects faces using Haar Cascade

Extracts face regions (ROI)

Trains an LBPH recognizer

Saves:

face_trained.yml – trained model

features.npy – face features

labels.npy – labels

📁 Project Structure
Celebrities/
│── haar_face.xml
│── face_train.py
│── face_trained.yml      (Generated after training)
│── features.npy          (Generated after training)
│── labels.npy            (Generated after training)

Faces/
|──train/
├── Person1/
├── img1.jpg
├── img2.jpg
├── Person2/
├── img1.jpg
├── img2.jpg
.....

🧠 How the Training Works

Load Haar Cascade

Loop through each person folder

Read images and convert to grayscale

Detect faces

Extract face ROI and save as feature

Generate numeric labels

Train LBPH recognizer

Save model + numpy arrays

🧪 Run Training Script
python face_train.py


📦 Model Output

face_trained.yml → Trained recognizer model

features.npy → Stored face regions

labels.npy → Numeric labels

You can later load this model for real-time face detection.
