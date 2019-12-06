#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle


DATA_DIR = "data/"
CATEGORIES = ["dog", "cat"]
IMG_SIZE = 50

training_data = []

# create datasets every image in CATEGORIES
# will saved eg. training_data([dog_array], [dog])
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# randomize the data order to make reduce model biases
random.shuffle(training_data)

# features and label
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# convert into array show can fit into tf
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# save data
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()


pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
