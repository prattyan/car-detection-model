# Car and License Plate Detection

This project contains three Python scripts that demonstrate **car detection** and **license plate recognition** using different approaches: TensorFlow, YOLO, and OpenCV with Tesseract OCR.

---

## 🚀 Features
1. **TensorFlow Object Detection**
   - Loads a pre-trained TensorFlow model.
   - Detects cars in images and draws bounding boxes.
   - Saves and displays the detection results.

2. **YOLO (You Only Look Once) Detection**
   - Uses YOLOv3 for real-time car detection in images and videos.
   - Draws bounding boxes only for cars.
   - Supports both image and video input.

3. **License Plate Recognition**
   - Uses OpenCV to detect license plates in car images.
   - Applies preprocessing and contour detection to locate plates.
   - Uses Tesseract OCR to extract text from detected plates.

---

## 📂 Files
- `car_detection_tensorflow.py` → Car detection with TensorFlow.
- `car_detection_yolo.py` → Car detection using YOLOv3.
- `license_plate_recognition.py` → License plate detection and OCR.

---

## ⚙️ Requirements
Install dependencies before running the scripts:

```bash
pip install tensorflow opencv-python numpy pytesseract
```

Additionally:
- Download **YOLOv3 weights** and **cfg file** from the [official YOLO website](https://pjreddie.com/darknet/yolo/).
- Place `yolov3.weights`, `yolov3.cfg`, and `coco.names` in the same directory.

For Tesseract OCR:
- Install Tesseract from [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract).
- Update the path in the script if needed (e.g., Windows requires `pytesseract.pytesseract.tesseract_cmd` to point to `tesseract.exe`).

---

## ▶️ Usage

### TensorFlow Car Detection
```bash
python car_detection_tensorflow.py
```
Modify the script to provide the image path.

### YOLO Car Detection (Image)
```bash
python car_detection_yolo.py
```
Edit the script with your image path.

### YOLO Car Detection (Video)
```bash
python car_detection_yolo.py
```
Press `q` to quit the video display.

### License Plate Recognition
```bash
python license_plate_recognition.py
```
Replace `car_image.jpg` with your input image.

---

## 📌 Notes
- Adjust confidence thresholds as needed.
- Ensure paths for models (`.weights`, `.cfg`, `.pb`) are correct.
- For real-time detection, use a webcam by setting `cv2.VideoCapture(0)` in the YOLO script.

---

## 📜 License
This project is for **educational purposes only**. Free to use and modify.

---

👨‍💻 Developed by **Prattyan Ghosh**
