# Facial Emotion Recognition (FER) Framework

## Overview

This repository implements a real-time Facial Emotion Recognition (FER) system based on two complementary approaches:

1. Landmark-based feature extraction using MediaPipe and Support Vector Machines (SVM)
2. Deep learning inference using TensorFlow Lite (TFLite)

The framework is designed for both high-performance environments and resource-constrained devices such as Raspberry Pi.

---

## Features

* Real-time face detection and emotion recognition
* Landmark-based FER using selected facial keypoints
* Deep learning FER using optimized TFLite models
* Webcam-based live inference
* Lightweight deployment capability
* Modular and extensible design

---

## Project Structure

```
├── Paper_L03_Test.py              # Landmark-based FER (SVM + MediaPipe)
├── Paper_MobileNet_Test_Big.py   # TFLite FER (higher accuracy)
├── Paper_MobileNet_Test_Mini.py  # Lightweight TFLite FER
├── Paper_Convertor.py            # Convert Keras (.h5) to TFLite
├── requirements.txt              # Python dependencies
├── setup_env.sh                  # Environment setup script
├── SVM.pkl                       # Trained SVM model
├── fer_model.tflite              # TFLite model
```

---

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
chmod +x setup_project.sh
./setup_project.sh
```

This script will:

* Clone the repository
* Create required directories (`models`, `outputs`, `logs`, `data`)
* Create a Python virtual environment
* Install all dependencies

---

### Option 2: Manual Setup

```bash
git clone https://github.com/hosamzolfonoon/Papep_FER_Test.git
cd Papep_FER_Test

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Methods

### 1. Landmark-Based FER (SVM)

This approach extracts a subset of facial landmarks from MediaPipe FaceMesh and applies normalization before classification.

Pipeline:

* Face detection and landmark extraction
* Selection of relevant landmark indices
* Coordinate normalization
* Emotion classification using SVM

Run:

```bash
python Paper_L03_Test.py
```

---

### 2. Deep Learning FER (TFLite)

This approach uses a convolutional neural network converted to TensorFlow Lite format for efficient inference.

Features:

* Quantized model support
* Real-time processing
* Optimized for embedded devices

Run:

```bash
python Paper_MobileNet_Test_Mini.py
```

or

```bash
python Paper_MobileNet_Test_Big.py
```

---

## Pre-trained Model Conversion

The framework supports external FER models such as:

* `fer2013_mini_XCEPTION.119-0.65.hdf5`
* `fer2013_big_XCEPTION.54-0.66.hdf5`

These models are available from:

[https://github.com/oarriaga/face_classification](https://github.com/oarriaga/face_classification)

Since this project relies on TensorFlow Lite for efficient inference, the downloaded `.hdf5` models must be converted to `.tflite` format before use.

To perform the conversion:

```bash
python Paper_Convertor.py
```

The conversion script loads the `.hdf5` model and exports an optimized `.tflite` version suitable for real-time execution.

After conversion:

* Place the generated `.tflite` file in the project directory or inside `models/`
* Update the model path in the inference scripts if necessary

---

## Usage

* Execute one of the provided scripts
* The webcam will open automatically
* Detected faces will be annotated with predicted emotions
* Press `q` to terminate execution

---

## Supported Emotion Classes

Typical emotion categories include:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## Performance Considerations

* The Mini model is suitable for low-resource devices
* The Big model provides higher accuracy at increased computational cost
* Reducing camera resolution can improve frame rate

---

## Reproducibility

* All dependencies are specified in `requirements.txt`
* Environment setup scripts are provided
* Modular execution allows independent evaluation
* Compatible with standard FER datasets such as FER2013, JAFFE, and KDEF

---

## Requirements

* Python 3.8 or higher
* Webcam device
* Optional: Raspberry Pi for edge deployment

---

## Citation

```bibtex
@article{zolfonoon2026fer,
  title={A Lightweight Vision-Based Emotion Sensing Framework for Assistive Healthcare Robotics},
  author={Zolfonoon, Hosam and Araújo, Helder and Marques, Lino},
  journal={Sensors},
  year={2026}
}
```

---

## Author

Hosam Zolfonoon
Institute of Systems and Robotics (ISR)
University of Coimbra

---

## License

This project is intended for research and academic use.


