# Face Detection & Gender Classification

## Project Description

This project performs **face detection** from images or video streams and then **classifies the gender** of each detected face (Man / Woman).

It combines a **YOLO face detection model** with a **Machine Learning classifier (Decision Tree)** trained on facial features extracted from images.

---

## What Does This Code Do?

1. Detects one or multiple faces in a single image or video frame
2. Crops each detected face
3. Extracts visual features from each face
4. Trains a Machine Learning model to recognize gender
5. Predicts the gender of each face in real-time

---

## Technologies Used
<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="40"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="40"/>
</p>

- ![YOLO](https://img.shields.io/badge/YOLOv9-Ultralytics-00FFFF?style=flat) – face detection
- ![Scikit-image](https://img.shields.io/badge/scikit--image-0C55A5?style=flat) – feature extraction (HOG, LBP, Gabor filters)
---



## Project Structure

```bash
project/
│
├── model/
│   └── yolov9m-face-lindevs.pt
│
├── assert/
│   ├── output1.png
│   └── output2.png
│
├── data-set/
│   ├── mans/
│   └── womans/
│   
├── src/
│   └── face_recognaion.py
├── requirments.txt
└── README.md
```

---

## Dataset

* `mans/` : contains images of male faces
* `womans/` : contains images of female faces

Each image may contain **one or multiple faces**.

---

## How It Works

### 1️⃣ Face Detection

#### Uses a pretrained **YOLOv9 face detection model** to locate faces in an image or video frame.

```python
faces = model(image_rgb)
```

---

### 2️⃣ Feature Extraction

#### For each detected face, the following features are extracted:

* **Gabor Filter** – texture & frequency information
* **Local Binary Pattern (LBP)** – texture patterns
* **Histogram of Oriented Gradients (HOG)** – shape & edge information

#### These features are concatenated into a single feature vector.

---

### 3️⃣ Model Training

#### A **Decision Tree Classifier** is trained using:

* **X** → extracted facial features
* **Y** → gender labels (`man`, `woman`)

```python
model_ML.fit(features, labels)
```

---

### 4️⃣ Gender Prediction

#### For a new image or video frame:

* Detect faces
* Extract features
* Predict gender for each face

```python
model_ML.predict(features)
```

---

## Video & Camera Support

* Supports webcam (`0`) or video file path
* Draws bounding boxes around faces
* Displays predicted gender on each face

#### Press **Q** to exit the video window.

---

## How to Run

1. Install required libraries:

```bash
pip install -r requirements.txt
```

2. Place the YOLO face model in the `model/` directory

3. Organize training images into:

```
data-set/mans/
data-set/womans/
```

4. Run the script:

```bash
python src/face_recognaion.py
```

5. Enter:

* `0` for webcam
* or absolute path to a video file

---

## Output Example

- White bounding boxes indicate detected faces
- Gender label displayed above the face
- Text label shows predicted gender for each face
- Below are example outputs of the system while running on images or video frames.

### Example:

### Sample Outputs

![Output Example 1](/assert/output1.png)
![Output Example 2](/assert/output2.png)


---

## Future Improvements

* Use CNN / Deep Learning instead of Decision Tree
* Improve accuracy with larger datasets
* Add age estimation
* Add face recognition (identity)

---

## Notes

* Accuracy depends on dataset quality
* Images should clearly show faces
* Multiple faces per image are supported

---

## Author

Developed for face detection and gender classification using computer vision and machine learning.
