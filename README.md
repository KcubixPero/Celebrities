<h1 align="center">Face Recognition Training Script</h1>

<p align="center">
  <strong>A Python + OpenCV project that trains an LBPH face recognizer using Haar Cascade detection.</strong>
</p>

<hr>

<h2>📁 Project Structure</h2>

<pre>
Faces/
 └── train/
      ├── Person1/
      ├── Person2/
      └── Person3/
Celebrities/
 ├── haar_face.xml
 ├── face_trained.yml
 ├── features.npy
 └── labels.npy
face_train.py
</pre>

<hr>

<h2>🚀 How It Works</h2>

<ul>
  <li>Reads images from <code>Faces/train</code></li>
  <li>Detects faces using <code>Haar Cascade</code></li>
  <li>Crops face regions and stores them as features</li>
  <li>Labels each face based on folder name</li>
  <li>Trains an <strong>LBPH Face Recognizer</strong></li>
  <li>Saves:</li>
  <ul>
    <li><code>face_trained.yml</code></li>
    <li><code>features.npy</code></li>
    <li><code>labels.npy</code></li>
  </ul>
</ul>

<hr>

<h2>📦 Requirements</h2>

<pre>
pip install opencv-python
pip install opencv-contrib-python
numpy
</pre>

<hr>

<h2>▶️ Running the Training Script</h2>

<pre>
python face_train.py
</pre>

<hr>

<p>This project is for educational purposes only.</p>
