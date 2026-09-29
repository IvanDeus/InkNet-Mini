# InkNet-Mini: Personalized Handwritten Digit Recognition from Minimal Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**InkNet-Mini** is a lightweight Convolutional Neural Network (CNN) experiment designed to learn and recognize a *single individual's* handwriting style using only **100 base images** (10 per digit). 

While standard digit recognition models rely on massive datasets like MNIST (60,000+ images), this project demonstrates how deep learning can be adapted for highly personalized, low-data scenarios. By leveraging aggressive data augmentation and a constrained network architecture, InkNet-Mini learns the unique quirks of your handwriting without needing thousands of examples.

## ✨ Features

- **Minimal Data Footprint:** Requires only 10 original images per digit (100 total).
- **Automated Data Augmentation:** Generates 20 unique variations per image (rotation, shifting, zooming, noise) to prevent overfitting.
- **Custom CNN Architecture:** Optimized for small datasets with strategic dropout layers.
- **Multi-Digit Inference:** Can process full phone numbers or postal codes, not just isolated digits.
- **Modular Pipeline:** Clean separation between data augmentation, model training, and inference.

## 📂 Project Structure

```text
InkNet-Mini/
├── augment_data.py          # Script to generate augmented training images
├── train_custom.py          # Script to train the CNN on the enhanced dataset
├── predict.py               # Inference script for single/multi-digit images
├── requirements.txt         # Python dependencies
├── custom_data/             # Your dataset directory
│   ├── 0/                   # Contains 00.png to 09.png (and generated 10.png-209.png)
│   ├── 1/
│   └── ...
└── my_handwriting_model.h5  # The trained model (generated after training)
```

## 🛠️ Prerequisites

- **OS:** Ubuntu 24.04 (or any modern Linux/macOS/Windows)
- **Python:** 3.10 or higher (Python 3.12 recommended for Ubuntu 24.04)
- **Hardware:** CPU is sufficient (GPU optional but faster)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IvanDeus/InkNet-Mini.git && cd InkNet-Mini
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage Guide

### Step 1: Prepare Your Base Dataset
Create the `custom_data` directory structure and add exactly **10 images** for each digit (0-9). 
- Name them `00.png` through `09.png` inside their respective folders.
- **Tip:** Write the digits slightly differently each time (vary the slant, size, and pen pressure) to give the model more natural variance to learn from.

```bash
mkdir -p custom_data/{0,1,2,3,4,5,6,7,8,9}
# Add your 100 images (10 per folder) here.
```

### Step 2: Augment the Data
Run the augmentation script to expand your 100 images into 2,100 images. This prevents the neural network from simply memorizing your original 100 pictures.

```bash
python augment_data.py
```
*You will now see `10.png` through `209.png` in each folder. Open a few to verify they look like slightly altered versions of your handwriting. Original images will be reformatted.*

### Step 3: Train the Model
Train the CNN on your newly expanded dataset. The script will automatically split the data into 80% training and 20% testing.

```bash
python train_custom.py
```
*Watch the console output. You want to see both `accuracy` (training) and `val_accuracy` (testing) climb together. This will generate `my_handwriting_model.h5`.*

### Step 4: Predict / Inference
Use the prediction script to test the model on brand new images of your handwriting.

**Recognize a single digit:**
```bash
python predict.py path/to/single_digit.png
```

**Recognize a sequence (like a phone number or postal code):**
```bash
python predict.py path/to/phone_number.png --multi
```

## 🧠 How It Works

1. **The Overfitting Problem:** If you train a deep learning model on only 10 images per class, it will achieve 100% training accuracy but fail completely on new data. It memorizes the exact pixels rather than learning the concept of the digit.
2. **The Augmentation Solution:** `augment_data.py` applies random affine transformations (rotations up to 15°, shifts, zooms, blur, and noise). This forces the model to learn the *structural features* of your handwriting (loops, lines, intersections) rather than pixel-perfect memorization.
3. **The Architecture:** The CNN uses `Dropout(0.5)` in its final dense layer. During training, this randomly turns off 50% of the neurons, further preventing the network from relying on any single memorized feature.
4. **Increase Accuracy:** If you decide you want higher accuracy for a practical application, do not increase augmented images, but start increasing your original images.

## 💡 Tips for Best Results

- **Contrast is King:** Ensure your original images have high contrast (black ink on white paper). The scripts automatically invert colors if needed, but clean source images yield the best results.
- **Centering:** Try to draw the digits roughly in the center of the image frame.
- **Isolate Digits for Multi-Prediction:** When using the `--multi` flag, ensure there is clear, empty space between each digit in the source image so the contour detection can separate them accurately.

---
2026 [ ivan deus ]
