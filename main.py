import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# Load your dataset
def load_data(data_dir):
    images = []
    labels = []
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            # For simplicity, assuming labels are stored as a tuple (age, sex, emotion) in a separate file or inferred
            labels.append(get_label_for_image(img_name))
    return np.array(images), np.array(labels)