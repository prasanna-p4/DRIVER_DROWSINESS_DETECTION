import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model(
    r"D:\DRIVER_DROWSINESS_DETECTION\drowsy_model.h5"
)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r"D:\DRIVER_DROWSINESS_DETECTION\dataset",
    target_size=(24, 24),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

# Evaluate accuracy
loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy * 100)
