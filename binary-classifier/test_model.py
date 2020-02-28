#!/bin/python3
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

CATEGORIES = ["dog", "cat"]


# proccess test img
def proccess_img(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array / 255.0
    reshaped_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped_array


# load model
model = tf.keras.models.load_model("binary_classifier.model")

# predict test
test_data = os.listdir("./test-data")

for test in test_data:
    predict = model.predict([proccess_img('test-data/' + test)])
    result = predict[0]

    # if statement because of 80% acc
    if result >= [0.9]:
        result = CATEGORIES[1]
    else:
        result = CATEGORIES[0]
    print(result, test)
