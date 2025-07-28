import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
import os
load_dotenv()

# Dataset directory
data_dir = os.getenv("dataset_path")+"/images"


# Data preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.4]
)

train_data = datagen.flow_from_directory(
    data_dir + "/train", target_size=(256, 256), batch_size=16, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(
    data_dir + "/val", target_size=(256, 256), batch_size=16, class_mode='binary', subset='validation')

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a modified learning rate


# Use a learning rate scheduler
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)

from tensorflow.keras.callbacks import EarlyStopping

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

import numpy as np

# Calculate class weights
class_weights = {0: len(train_data.classes) / (2 * np.bincount(train_data.classes))[0],
                 1: len(train_data.classes) / (2 * np.bincount(train_data.classes))[1]}

print(f"Class Weights: {class_weights}")

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Train the model with the new architecture and learning rate scheduler
model.fit(train_data, validation_data=val_data, epochs=30, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights)
model.save(os.getenv("model_path")+"/fall_detection_model.h5")

print("Model trained and saved successfully!")