# Accident-Detection-Using-AI-ML
# Accident Detection and Alert System

## Overview
This application uses machine learning to detect accidents in videos and live camera feeds. When an accident is detected, the system can automatically send SMS alerts to emergency contacts using Twilio.

## Features
- **Video-Based Accident Detection**: Upload and analyze video files for accidents
- **Camera-Based Real-time Detection**: Use a webcam for live accident detection
- **Auto-Pause on Accident**: Video playback pauses when an accident is detected
- **SMS Alerts**: Sends automatic SMS notifications when accidents are detected
- **Modern Web Interface**: Easy-to-use Streamlit interface

## Technologies Used
- **TensorFlow**: Deep learning framework for accident detection model
- **OpenCV**: Computer vision library for video processing
- **Streamlit**: Web interface for user interaction
- **Twilio**: SMS notification service

## Requirements
- Python 3.7+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository or extract the ZIP file
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Method 1: Using Streamlit (Recommended)
Run the application with Streamlit for a modern web interface:
```bash
cd "Accident Detection with UI"
streamlit run app.py
```

### Method 2: Using Traditional UI
Run the application with the Tkinter-based interface:
```bash
cd "Accident Detection with UI"
python main.py
```

## How to Use

### Video Detection
1. Navigate to "Accident Detection (Video)" in the sidebar
2. Upload a video file (.mp4, .avi, or .mkv)
3. Adjust the detection threshold if needed
4. Click "Start Processing"
5. The video will automatically pause when an accident is detected
6. Click "Continue" to resume processing

### Camera Detection
1. Navigate to "Accident Detection (Camera)" in the sidebar
2. If in a cloud environment without camera access, check "Use test video instead of camera"
3. Click "Start Camera/Video" to begin detection
4. Click "Stop" to end the detection

### SMS Alerts
To receive SMS alerts:
1. Enter your phone number including country code in the input field
2. Make sure your Twilio account is correctly configured in the code

## How It Works

### Technical Details
1. **Video Processing**: The application reads video frames using OpenCV
2. **Image Preprocessing**: Frames are resized to 224x224 and converted to grayscale
3. **Accident Detection**: A pre-trained deep learning model analyzes each frame
4. **Alert System**: When an accident is detected, SMS alerts are sent via Twilio

### Machine Learning Model
The project uses a convolutional neural network trained on accident footage. The model is stored in the "model/model.h5" file and expects grayscale images of size 224x224 pixels.

## Troubleshooting
- **Cannot access camera**: If running in a cloud environment, use the "Use test video instead of camera" option
- **Slow processing**: Adjust the frame processing rate in the code if needed
- **SMS not sending**: Verify your Twilio credentials and ensure you have sufficient credit

## License
This project is available for educational and research purposes.

## Acknowledgments
- The accident detection model is trained on a specialized dataset of traffic incidents
- Thanks to all libraries and frameworks that made this project possible
