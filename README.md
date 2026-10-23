# The Manipulation of Machine Learning Through Images

An interactive dashboard for demonstrating how image manipulations and adversarial attacks (FGSM) affect pre-trained Convolutional Neural Networks (CNNs), and how defense mechanisms like spatial filtering can recover predictions.

This project implements the system described in the Phase-I project report, using a modern tech stack (Python, PyTorch, Streamlit).

## Features

- **Upload & Classify:** Upload any image (JPG, PNG, WebP) and classify it using pre-trained ResNet50 or MobileNetV2 models (ImageNet labels).
- **Benign Manipulations:** Apply visual transformations like brightness, contrast, rotation, Gaussian noise, and JPEG compression.
- **Adversarial Attacks:** Execute the Fast Gradient Sign Method (FGSM) attack with a tunable epsilon ($\epsilon$) parameter to generate adversarial perturbations.
- **Defense Mechanisms:** Apply spatial filtering (Median Filter, Gaussian Blur, Bilateral Filter) to attempt to clean the manipulated images and recover the original classification.
- **Interactive Dashboard:** Compare the original, manipulated, and defended images side-by-side, along with their top-5 predictions and confidence scores.
- **Visual Analytics:** View grouped bar charts tracking the shift in prediction confidence across the pipeline.

## Project Structure

```text
The Manipulation of Machine Learning Through Images/
├── .streamlit/
│   └── config.toml          # Custom dark, premium UI theme configuration
├── models/
│   ├── classifier.py        # Core ML logic (loading ResNet/MobileNet, preprocessing, inference)
│   └── labels.py            # ImageNet-1K labels (dynamically loaded from torchvision)
├── utils/
│   ├── manipulation.py      # Implementations of benign transforms and the FGSM attack
│   └── defense.py           # Implementations of spatial filtering defenses
├── app.py                   # Main Streamlit interactive dashboard
└── requirements.txt         # Project dependencies
```

## Setup & Installation

### Prerequisites

Ensure you have Python 3.11 or later installed on your system.

### 1. Create a Virtual Environment (Recommended)

It's highly recommended to run this project inside a virtual environment to manage dependencies cleanly.

Open your terminal (or PowerShell) and navigate to the project directory:

```bash
cd "c:\Users\hp\OneDrive\Documents\The Manipulation of Machine Learning Through Images"
```

Create the virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

- On **Windows**:
  ```powershell
  .\venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies

Once the virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch installation might take a few minutes depending on your internet connection).*

## Running the Application

To start the interactive dashboard, run the following command from the project root directory:

```bash
streamlit run app.py
```

Streamlit will start the local server and automatically open the application in your default web browser at `http://localhost:8501`.

**Important Note:** The first time you run the application and select a model (ResNet50 or MobileNetV2), it will download the pre-trained weights from the internet (approximately 100MB for ResNet50). Subsequent runs will load the cached weights instantly.

## Usage Guide

1. **Configure:** Use the sidebar to select the CNN model (ResNet-50 or MobileNet V2).
2. **Upload:** Drag and drop an image into the upload area in the main dashboard. Wait for the baseline classification to complete.
3. **Attack / Manipulate:** In the sidebar, choose either a "Benign Transform" or the "FGSM Adversarial Attack". Adjust the parameters (like the $\epsilon$ slider for FGSM).
4. **Defend:** Check the "Apply Defense" box and select a spatial filter (like the 3x3 Median Filter) to see if the model can recover the correct prediction.
5. **Analyze:** Scroll down to view the side-by-side images, confidence drop/recovery metrics, and the interactive comparison chart.
