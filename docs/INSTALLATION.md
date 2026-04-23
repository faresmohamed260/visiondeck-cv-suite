# Installation Guide

## Requirements

- Windows 10 or newer
- Python 3.10 or newer
- Webcam or Android phone with **IP Webcam**

## 1. Clone the Repository

```bash
git clone https://github.com/faresmohamed260/visiondeck-cv-suite.git
cd visiondeck-cv-suite
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Virtual Environment

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
.venv\Scripts\activate.bat
```

## 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Run the App

```bash
python -m streamlit run streamlit_app.py
```

## 6. Open the Dashboard

Streamlit will print a local address such as:

```text
http://localhost:8501
```

Open it in your browser.

## Troubleshooting

### Camera is not detected

- Click `Refresh Cameras`
- Make sure no other app is locking the camera
- Reconnect the phone and laptop to the same Wi-Fi network

### IP Webcam is not found

- Start the IP Webcam server on the Android device
- Ensure both devices are on the same network
- Refresh the dashboard again after the server starts

### Slow performance

- Prefer the laptop webcam for lower latency
- Use a stable Wi-Fi connection for IP Webcam
- Close unnecessary apps using the camera or CPU
