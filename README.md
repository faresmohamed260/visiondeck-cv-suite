# VisionDeck

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VisionDeck is a lightweight computer vision dashboard that bundles three academic mini-projects into one polished Streamlit app:

- Face Detection System
- Hand Gesture Recognition
- Real-Time Object Detection

It supports both laptop webcams and Android phones running the **IP Webcam** app, with automatic IP camera discovery on the local network.

## Features

- Simple Streamlit dashboard for running all three projects
- Separate project modules for clean coursework organization
- Support for local webcam and Android IP Webcam
- Automatic IP camera discovery on the same network
- Live video feed inside the dashboard
- Image upload support for testing still images
- Face landmarks and face bounding boxes
- Hand skeleton overlay with gesture recognition
- Real-time object detection with YOLO

## Demo Modules

### 1. Face Detection System

- Detects multiple faces in a frame
- Draws bounding boxes around faces
- Highlights facial landmark points

### 2. Hand Gesture Recognition

- Tracks hands in real time
- Draws hand landmarks and connections
- Recognizes common gestures such as:
  - Open Palm
  - Fist
  - Peace
  - Pointing
  - Thumbs Up
  - Thumbs Down

### 3. Real-Time Object Detection

- Detects multiple objects in a frame
- Draws class labels and bounding boxes
- Uses a lightweight YOLO model for practical real-time performance

## Tech Stack

- Python 3.10+
- Streamlit
- OpenCV
- MediaPipe
- Ultralytics YOLO
- Requests

## Project Structure

```text
visiondeck-cv-suite/
├── common/
│   └── camera.py
├── face_detection_system/
│   └── detector.py
├── hand_gesture_recognition/
│   └── recognizer.py
├── real_time_object_detection/
│   └── detector.py
├── docs/
│   └── INSTALLATION.md
├── requirements.txt
└── streamlit_app.py
```

## Quick Start

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Run the Streamlit dashboard.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Camera Setup

### Laptop Webcam

- Open the dashboard
- Click `Refresh Cameras`
- Select the detected local camera
- Choose a project and click `Start`

### Android IP Webcam

1. Install the **IP Webcam** app on your Android phone.
2. Connect the phone and laptop to the same Wi-Fi network.
3. Start the server from the app.
4. In VisionDeck, click `Refresh Cameras`.
5. Select the discovered `IP Webcam (...)` source.

## Documentation

- Installation guide: [docs/INSTALLATION.md](docs/INSTALLATION.md)
- Repository: [github.com/faresmohamed260/visiondeck-cv-suite](https://github.com/faresmohamed260/visiondeck-cv-suite)

## Notes

- The first object detection run may download `yolov8n.pt`.
- For best performance, use good lighting and a stable camera angle.
- IP camera performance depends on Wi-Fi quality.

## License

This project is released under the [MIT License](LICENSE).
