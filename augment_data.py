# augment_data.py
import os
import cv2
import numpy as np

def preprocess_to_mnist(img):
    """Convert image to 28x28, pure white text on pure black background"""
    # 1. Force exact same size (28x28)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 2. Smart Inversion: Check the corners to determine background color
    # If corners are bright, it's white paper with black ink -> invert it.
    corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    if np.mean(corners) > 127:
        img = 255 - img
        
    # 3. Strict Thresholding: Remove paper texture, shadows, and gray backgrounds.
    # Forces background to pure black (0) and ink to pure white (255).
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
    return img

def augment_image(img):
    """Apply random transformations to create a new variation"""
    h, w = img.shape[:2]
    
    # 1. Random Rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h), borderValue=0) 
    
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
        
    # 5. FIXED Noise Addition: Apply noise ONLY to the white ink, not the black background
    if np.random.rand() > 0.5:
        # Create a mask of the ink (pixels > 0)
        mask = img > 0 
        # Generate float noise to prevent uint8 wrap-around bug
        noise = np.random.normal(0, 15, img.shape) 
        
        # Apply noise only to the ink pixels
        img_float = img.astype(np.float32)
        img_float[mask] += noise[mask]
        
        # Clip and convert back to uint8
        img = np.clip(img_float, 0, 255).astype(np.uint8)
        
        # Re-threshold to ensure blur/noise didn't create gray pixels in the background
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
    return img

def main():
    base_dir = 'custom_data'
    num_augmentations_per_image = 30  # Generates x new images
    
    print("Starting data augmentation and preprocessing...")
    
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
                
            # PREPROCESS THE ORIGINAL IMAGE
            img = preprocess_to_mnist(img)
            
            # OVERWRITE THE ORIGINAL FILE with the cleaned-up version
            cv2.imwrite(img_path, img)
            
            # Generate and save augmented variations based on the cleaned image
            for i in range(num_augmentations_per_image):
                aug_img = augment_image(img.copy())
                aug_name = f"{aug_count:02d}.png"
                cv2.imwrite(os.path.join(folder, aug_name), aug_img)
                aug_count += 1
                
        print(f"Digit {digit}: Preprocessed & overwrote {len(original_files)} originals, generated {aug_count - 10} augmented images.")
        
    print("\nDone! All images are now pure black/white, correctly sized, and cleanly augmented.")

if __name__ == "__main__":
    main()
