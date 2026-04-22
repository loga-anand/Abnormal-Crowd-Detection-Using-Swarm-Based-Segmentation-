import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model once
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_cnn_features(image):
    """
    image: NumPy array (H, W, 3)
    returns: feature vector (1280,)
    """

    # Ensure 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Resize
    image = tf.image.resize(image, (224, 224))
    image = image.numpy().astype("float32")

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Preprocess
    image = preprocess_input(image)

    # Extract features
    features = base_model.predict(image, verbose=0)

    return features.flatten()
