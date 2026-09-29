# predict.py
import cv2
import numpy as np
import tensorflow as tf
import sys
import os

def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

def preprocess_digit(digit_img):
    """Preprocess single digit to match MNIST format"""
    # Convert to grayscale if needed
    if len(digit_img.shape) == 3:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    
    # Invert if black on white (MNIST is white on black)
    if np.mean(digit_img) > 127:
        digit_img = 255 - digit_img
    
    # Resize to 20x20 (MNIST standard)
    digit_img = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 canvas with digit centered
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = digit_img
    
    # Normalize
    canvas = canvas.astype('float32') / 255.0
    
    return canvas.reshape(1, 28, 28, 1)

def predict_single_digit(model, img_path):
    """Predict single digit from image file"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    processed = preprocess_digit(img)
    prediction = model.predict(processed, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return digit, confidence

def predict_multiple_digits(model, img_path):
    """Predict multiple digits from image (e.g., phone number)"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    # Threshold to binary
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (individual digits)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "No digits found", []
    
    # Sort contours left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_with_boxes = list(zip(contours, bounding_boxes))
    contours_with_boxes.sort(key=lambda x: x[1][0])  # Sort by x coordinate
    
    digits = []
    confidences = []
    
    for contour, (x, y, w, h) in contours_with_boxes:
        # Filter out noise (too small or too large)
        if w < 5 or h < 5 or w > img.shape[1] * 0.8:
            continue
        
        # Extract digit region with padding
        padding = 5
        y1 = max(0, y - padding)
        y2 = min(img.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(img.shape[1], x + w + padding)
        
        digit_img = img[y1:y2, x1:x2]
        
        # Predict
        processed = preprocess_digit(digit_img)
        prediction = model.predict(processed, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        digits.append(str(digit))
        confidences.append(confidence)
    
    return ''.join(digits), confidences

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single digit:  python predict.py image.png")
        print("  Multiple:      python predict.py image.png --multi")
        sys.exit(1)
    
    img_path = sys.argv[1]
    multi_mode = '--multi' in sys.argv
    
    if not os.path.exists('mnist_model.h5'):
        print("Error: mnist_model.h5 not found. Run train.py first.")
        sys.exit(1)
    
    model = load_model()
    
    if multi_mode:
        result, confidences = predict_multiple_digits(model, img_path)
        print(f"Recognized: {result}")
        if confidences:
            print("Confidences:", [f"{c:.2%}" for c in confidences])
    else:
        digit, confidence = predict_single_digit(model, img_path)
        print(f"Recognized: {digit}")
        print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()
