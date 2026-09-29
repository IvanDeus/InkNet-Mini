# train_custom.py
import os
import cv2
import numpy as np
import tensorflow as tf

def load_enhanced_dataset():
    X = []
    y = []
    
    print("Loading enhanced dataset...")
    for digit in range(10):
        folder = f'custom_data/{digit}'
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} not found. Run augment_data.py first.")
            
        files = [f for f in os.listdir(folder) if f.endswith('.png')]
        for f in files:
            img_path = os.path.join(folder, f)
            # Images are already 28x28 and white-on-black from the augmentation script
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            X.append(img)
            y.append(digit)
            
    # Normalize pixel values to 0-1
    X = np.array(X).reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y = np.array(y)
    
    return X, y

def main():
    X, y = load_enhanced_dataset()
    print(f"Total images loaded: {len(X)}")
    
    # Shuffle the data so training and test sets have a mix of all digits
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into 80% training, 20% testing
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training set: {len(X_train)} images | Testing set: {len(X_test)} images")
    
    # Build a slightly more robust model since we now have ~1680 training images
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout prevents overfitting
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train, 
        epochs=20, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save the model
    model.save('my_handwriting_model.h5')
    print("\nModel saved to my_handwriting_model.h5")
    
    # Final evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Test Accuracy on unseen augmented data: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
