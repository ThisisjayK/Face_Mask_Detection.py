 Face Mask Detection System using Raspberry Pi & Arduino

## 🧠 Overview

This project implements a real-time **Face Mask Detection and Temperature Screening System** using **Raspberry Pi**, **Arduino**, and **deep learning**. It leverages computer vision to determine if a person is wearing a face mask and checks body temperature using a contactless sensor. Based on the results, it triggers access control mechanisms such as a servo door motor and buzzer alerts.

---

## 📸 Features

- Real-time face mask detection using OpenCV and TensorFlow/Keras.
- Lightweight CNN model built on MobileNetV2.
- Serial communication between Raspberry Pi and Arduino for control logic.
- Buzzer alerts for people not wearing masks.
- Automatic door control using a servo motor if mask is detected and temperature is normal.
- Contactless body temperature measurement using MLX90614 IR sensor.
- Compact, deployable system suitable for public spaces like hospitals, banks, offices, and malls.

---

## 🛠 Hardware Components

- Raspberry Pi 4 Model B (with camera or USB webcam)
- Arduino Uno
- Pi Camera Module / USB Webcam
- MLX90614 IR Temperature Sensor
- Servo Motor (for door control)
- Buzzer
- LED (optional for visual alert)
- LCD (optional for status display)
- Jumper wires, breadboard, power supply

---

## 💻 Software & Libraries

- Python 3
- OpenCV
- TensorFlow
- Keras
- NumPy
- PySerial (for Arduino communication)
- Arduino IDE

---

## 🧾 Folder Structure

```
face_mask_detector_project/
├── detect_mask_pi.py              # Main Python script for Raspberry Pi
├── arduino_alert.ino              # Arduino sketch for buzzer control
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and setup
├── mask_detector.model/           # Pre-trained face mask detection model (to be downloaded)
└── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
```

---

## 🧪 Setup Instructions

### 1. Raspberry Pi
- Install required packages:
```bash
pip3 install -r requirements.txt
```
- Connect the Pi Camera or USB webcam.
- Download and place the trained model `mask_detector.model` into the project directory.
- Run the script:
```bash
python3 detect_mask_pi.py
```

### 2. Arduino
- Connect buzzer to pin 13.
- Upload `arduino_alert.ino` using Arduino IDE.
- Connect Arduino to Pi via USB.

---

## 🖼 Output

- Green box = Mask detected
- Red box = No mask detected (buzzer will sound)
- Servo door opens for compliant users
- Non-contact temperature check via IR sensor
- Real-time monitoring on Pi display

---

## 🧰 Applications

- Hospitals and Clinics
- Airports and Metro Stations
- Offices and Workspaces
- Shopping Malls and Banks
- Schools and Colleges

---

## 🚀 Future Enhancements

- Add voice alerts using speaker module.
- Store violation logs with timestamps.
- Send notifications via email or SMS.
- Integrate face recognition for authorized access.

---

## 📃 Credits

Developed by:
- D. Akhilesh
- J. Harsha
- K. Rohith Reddy
- K. Jayanth Adithya

Guided by:  
Mr. L. Amarender Reddy, Assistant Professor, KMIT Hyderabad
