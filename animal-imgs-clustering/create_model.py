#!/bin/python3
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import keras
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
import pickle


# load data
X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))


# model
vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

# transform 3d vectors from model to flat
def convnet_transform(covnet_model, raw_images):
    # pass training data to nn and flatten
    pred = covnet_model.predict(raw_images)
    flat = pred.reshape(raw_images.shape[0], -1)

    return flat

vgg16_output = covnet_transform(vgg16_model, X_train)
print(f"VGG16 flattened output has {vgg16_output.shape[1]} features")
vgg19_output = covnet_transform(vgg19_model, X_train)
print(f"VGG19 flattened output has {vgg19_output.shape[1]} features")
resnet50_output = covnet_transform(resnet50_model, X_train)
print(f"ResNet50 flattened output has {resnet50_output.shape[1]} features")
