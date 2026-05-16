import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Paths
train_path = "dataset/train"
valid_path = "dataset/valid"
test_path = "dataset/test"

# Image Settings
IMG_SIZE = 224
BATCH_SIZE = 32

# Training Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/Test Generator
valid_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load Training Data
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load Validation Data
valid_data = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load Test Data
test_data = valid_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Dataset Loaded Successfully!")


from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

# Load Pretrained EfficientNetB4
base_model = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build Final Model
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(15, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show Model Summary
model.summary()


# Train Model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=1
)

# Save Model
model.save("models/plant_disease_model.h5")

print("Model Saved Successfully!")
