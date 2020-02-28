#!/bin/python3
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import os, os.path
import random
import cv2
import glob
import pickle

DIR = "./animal_images"


def dataset_stats():
    animal_characters = ['C', 'D']
    stats = []

    for animal in animal_characters:
        # get a list of subdirectories
        directory_list = sorted(glob.glob(f"{DIR}/[{animal}]*"))

        for sub_directory in directory_list:
            file_names = [file for file in os.listdir(sub_directory)]
            file_count = len(file_names)
            sub_directory_name = os.path.basename(sub_directory)
            stats.append({
                "Code":
                sub_directory_name[:sub_directory_name.find('-')],
                "Image count":
                file_count,
                "Folder name":
                os.path.basename(sub_directory),
                "File names":
                file_names
            })

    df = pd.DataFrame(stats)
    return df


# print animal_images stats
dataset = dataset_stats().set_index("Code")
print(dataset[["Folder name", "Image count"]])


def load_images(codes):
    images = []
    labels = []

    for code in codes:
        folder_name = dataset.loc[code]["Folder name"]

        for file in dataset.loc[code]["File names"]:
            file_path = os.path.join(DIR, folder_name, file)
            # read the image using CV2
            image = cv2.imread(file_path)
            # resize
            image = cv2.resize(image, (224, 224))
            # convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(image)
            labels.append(code)
    return images, labels


# picking 4 animals breeds
codes = ["C1", "C8", "D11", "D25"]
images, labels = load_images(codes)


def show_random_images(images, labels, number_of_images_to_show=2):
    for code in list(set(labels)):
        indicies = [i for i, label in enumerate(labels) if label == code]
        random_indicies = [
            random.choice(indicies) for i in range(number_of_images_to_show)
        ]
        figure, axis = plt.subplots(1, number_of_images_to_show)

        print(f"{number_of_images_to_show} random images for code {code}")

        for image in range(number_of_images_to_show):
            axis[image].imshow(images[random_indicies[image]])
        plt.show()


show_random_images(images, labels)


def normalize_images(images, labels):
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    images /= 255

    return images, labels


images, labels = normalize_images(images, labels)


# shuffle data for training
def shuffle_data(images, labels):
    # set aside the testing data for test at the end
    X_train, X_test, y_train, y_test = train_test_split(images,
                                                        labels,
                                                        random_state=728)

    return X_train, y_train


X_train, y_train = shuffle_data(images, labels)
print(X_train)
print(y_train)

# save data
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()
