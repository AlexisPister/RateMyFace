from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import h5py

# Path of weights
model_weights_path = "./model_weights.h5"

# Model initialisation : resnet50 + denser layer with one cell (output) for regression
print('initializing model')
resnet = ResNet50(include_top=False, pooling="avg")
model = Sequential()
model.add(resnet)
model.add(Dense(1))

# Loading weights
print('loading weights')
model.load_weights(model_weights_path)

#%%
test = "./Dujardin.jpg"
img = cv2.imread(test)
img_proc = cv2.resize(img, (350, 350))

result = model.predict(np.array([img_proc]))

print("result : ", result)
plt.imshow(img)
plt.imshow(img_proc)
