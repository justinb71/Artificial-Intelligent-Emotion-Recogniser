from ast import Expression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop


pictureSize = 48
folderPath = "images/"

expression = "disgust"

plt.figure(figsize = (12, 12))

# for i in range(1, 10, 1):

#     plt.subplot(3, 3, i)
#     img = load_img(folderPath + "train/" + expression + "/" + os.listdir(folderPath + "train/" + expression)[i], target_size = (pictureSize, pictureSize))
#     plt.imshow(img)
# plt.show()



batchSize = 128

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

training_set = datagen_train.flow_from_directory(folderPath + "train", target_size = (pictureSize, pictureSize), color_mode= "grayscale", batch_size = batchSize, class_mode = "categorical", shuffle = True)

testing_set = datagen_train.flow_from_directory(folderPath + "validation", target_size = (pictureSize, pictureSize), color_mode= "grayscale", batch_size = batchSize, class_mode = "categorical", shuffle = True)


numberOfClasses = 7

model = Sequential()

#First CNN layer

model.add(Conv2D(64, (3,3), padding = "same", input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#Second CNN layer
model.add(Conv2D(128, (5, 5),padding = "same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

#Third CNN layer

mod
