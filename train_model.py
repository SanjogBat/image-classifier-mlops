import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Dataset: Intel Image Classification (Buildings, Forest, Glacier, Mountain, Sea, Street)
# Download from Kaggle and extract to a 'dataset' folder in this directory.
BATCH_SIZE = 32
IMAGE_SIZE = (150, 150)
EPOCHS = 10
DATASET_DIR = './dataset/seg_train/seg_train'

print("Loading dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# AUTOTUNE preloads images into RAM dynamically to prevent GPU starvation
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Building CNN Architecture...")
model = models.Sequential([
    # Input layer and normalization (Rescaling pixels from 0-255 to 0-1)
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flattening and Dense Classification Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(6, activation='softmax') # 6 classes (Softmax for probabilities)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# We save this in the specific format required by our Streamlit app
os.makedirs('models', exist_ok=True)
model.save('models/cnn_classifier.h5')
print("Model successfully saved to models/cnn_classifier.h5")
