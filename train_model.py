import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# CNN MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# DATASET LOADING
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'D:\DRIVER_DROWSINESS_DETECTION\dataset',
    target_size=(24, 24),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'D:\DRIVER_DROWSINESS_DETECTION\dataset',
    target_size=(24, 24),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("drowsy_model.h5")

print("✅ Model training completed & saved successfully")
