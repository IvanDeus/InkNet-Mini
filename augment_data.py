# augment_data.py
import os
import cv2
import numpy as np

def preprocess_to_mnist(img):
    """Convert image to 28x28, white text on black background (MNIST standard)"""
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # If image is mostly white (black pen on white paper), invert it
    if np.mean(img) > 127:
        img = 255 - img
        
    return img

def augment_image(img):
    """Apply random transformations to create a new variation"""
    h, w = img.shape[:2]
    
    # 1. Random Rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h), borderValue=0) # borderValue=0 keeps background black
    
    # 2. Random Translation (Shift X and Y)
    tx = np.random.uniform(-w * 0.15, w * 0.15)
    ty = np.random.uniform(-h * 0.15, h * 0.15)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (w, h), borderValue=0)
    
    # 3. Random Zoom/Scale (0.85x to 1.15x)
    scale = np.random.uniform(0.85, 1.15)
    M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    img = cv2.warpAffine(img, M_scale, (w, h), borderValue=0)
    
    # 4. Slight Gaussian Blur (simulates ink bleed/thick pen)
    if np.random.rand() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    # 5. Add slight noise (simulates paper texture/rough pen)
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        # Clip values to ensure they stay within 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    return img

def main():
    base_dir = 'custom_data'
    num_augmentations_per_image = 20  # Generates 20 new images per original
    
    print("Starting data augmentation...")
    
    for digit in range(10):
        folder = os.path.join(base_dir, str(digit))
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found. Skipping.")
            continue
            
        # Get original files (00.png to 09.png)
        original_files = sorted([f for f in os.listdir(folder) if f.endswith('.png') and int(f.split('.')[0]) < 10])
        
        if len(original_files) != 10:
            print(f"Warning: Expected 10 original images in {folder}, found {len(original_files)}.")

        aug_count = 10  # Start naming augmented files from 10.png
        
        for f in original_files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Error reading {img_path}")
                continue
                
            # Preprocess the original image first
            img = preprocess_to_mnist(img)
            
            # Generate and save augmented variations
            for i in range(num_augmentations_per_image):
                aug_img = augment_image(img.copy())
                aug_name = f"{aug_count:02d}.png"
                cv2.imwrite(os.path.join(folder, aug_name), aug_img)
                aug_count += 1
                
        print(f"Digit {digit}: Processed {len(original_files)} originals, generated {aug_count - 10} augmented images.")
        
    print("\nAugmentation complete! Your custom_data folders now contain 210 images each.")

if __name__ == "__main__":
    main()
