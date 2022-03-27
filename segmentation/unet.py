#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import os
import shutil
import re
import math
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_float32, img_as_ubyte
import pickle
import cv2

from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma
)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    print("gpus existem")
    print(gpus)
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class AugmentationSequence(keras.utils.Sequence):
  def __init__(self, x_set, y_set, batch_size, augmentations):
    self.x, self.y = x_set, y_set
    self.batch_size = batch_size
    self.augment = augmentations

  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    aug_x = np.zeros(batch_x.shape)
    aug_y = np.zeros(batch_y.shape)
    
    for idx in range(batch_x.shape[0]):
      aug = self.augment(image = batch_x[idx,:,:,:], mask = batch_y[idx,:,:,:])
      aug_x[idx,:,:,:] = aug["image"]
      aug_y[idx,:,:,:] = aug["mask"]
    
    return aug_x, aug_y



  # LOSS Functions
def jaccard_distance_loss(y_true, y_pred, smooth = 100):
      intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis = -1)
      union = keras.backend.sum(keras.backend.abs(y_true) + keras.backend.abs(y_pred), axis = -1)
      jac = (intersection + smooth) / (union - intersection + smooth)
      loss = (1 - jac) * smooth
      return loss

def dice_coef(y_true, y_pred, smooth = 1):
      intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis = -1)
      union = keras.backend.sum(keras.backend.abs(y_true), -1) + keras.backend.sum(keras.backend.abs(y_pred), -1)
      return (2. * intersection + smooth) / (union + smooth)


  # ## UNet

def unet_model(img_size):
    
    input_img = keras.layers.Input((img_size, img_size, 1), name = "img")
    
    # Contract #1
    c1 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(input_img)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Activation("relu")(c1)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c1)
    c1 = keras.layers.BatchNormalization()(c1)
    c1 = keras.layers.Activation("relu")(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    # Contract #2
    c2 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p1)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Activation("relu")(c2)
    c2 = keras.layers.Dropout(0.2)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c2)
    c2 = keras.layers.BatchNormalization()(c2)
    c2 = keras.layers.Activation("relu")(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Contract #3
    c3 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p2)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Activation("relu")(c3)
    c3 = keras.layers.Dropout(0.3)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c3)
    c3 = keras.layers.BatchNormalization()(c3)
    c3 = keras.layers.Activation("relu")(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    # Contract #4
    c4 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p3)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Activation("relu")(c4)
    c4 = keras.layers.Dropout(0.4)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c4)
    c4 = keras.layers.BatchNormalization()(c4)
    c4 = keras.layers.Activation("relu")(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    
    # Middle
    c5 = keras.layers.Conv2D(256, (3, 3), kernel_initializer = "he_uniform", padding = "same")(p4)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Activation("relu")(c5)
    c5 = keras.layers.Dropout(0.5)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c5)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Activation("relu")(c5)
    
    # Expand (upscale) #1
    u6 = keras.layers.Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = "same")(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u6)
    c6 = keras.layers.BatchNormalization()(c6)
    c6 = keras.layers.Activation("relu")(c6)
    c6 = keras.layers.Dropout(0.5)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c6)
    c6 = keras.layers.BatchNormalization()(c6)
    c6 = keras.layers.Activation("relu")(c6)
    
    # Expand (upscale) #2
    u7 = keras.layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = "same")(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u7)
    c7 = keras.layers.BatchNormalization()(c7)
    c7 = keras.layers.Activation("relu")(c7)
    c7 = keras.layers.Dropout(0.5)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c7)
    c7 = keras.layers.BatchNormalization()(c7)
    c7 = keras.layers.Activation("relu")(c7)
    
    # Expand (upscale) #3
    u8 = keras.layers.Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = "same")(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u8)
    c8 = keras.layers.BatchNormalization()(c8)
    c8 = keras.layers.Activation("relu")(c8)
    c8 = keras.layers.Dropout(0.5)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c8)
    c8 = keras.layers.BatchNormalization()(c8)
    c8 = keras.layers.Activation("relu")(c8)
    
    # Expand (upscale) #4
    u9 = keras.layers.Conv2DTranspose(16, (3, 3), strides = (2, 2), padding = "same")(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(u9)
    c9 = keras.layers.BatchNormalization()(c9)
    c9 = keras.layers.Activation("relu")(c9)
    c9 = keras.layers.Dropout(0.5)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), kernel_initializer = "he_uniform", padding = "same")(c9)
    c9 = keras.layers.BatchNormalization()(c9)
    c9 = keras.layers.Activation("relu")(c9)
    
    output = keras.layers.Conv2D(1, (1, 1), activation = "sigmoid")(c9)
    model = keras.Model(inputs = [input_img], outputs = [output])
    return model


def main():
  # ## Import Dataset


  img_size = 400
  #file_pi = open('input/train_dataset.pkl', 'rb') 
  #train_generator =  pickle.load(file_pi)

  X_train = np.load('input/X_train.npy')
  Y_train = np.load('input/Y_train.npy')

  X_val = np.load('input/X_val.npy')
  Y_val = np.load('input/Y_val.npy')

  with open('input/epochs.txt', 'r') as file:
      steps_per_epoch = file.read().rstrip()

  steps_per_epoch = 2* int(steps_per_epoch)

  print(steps_per_epoch)


  # ## Learning rate + Fit


  reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss", 
    factor = 0.5, 
    patience = 3, 
    verbose = 1
  )

  reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
  
  reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-7)

  checkpointer = keras.callbacks.ModelCheckpoint(
    "unet.h5", 
    verbose = 1, 
    save_best_only = True
  )


  #strategy = tf.distribute.MirroredStrategy()

  if (os.path.exists("unet.h5")):
      model = keras.models.load_model("unet.h5",
      custom_objects = {
        "jaccard_distance_loss": jaccard_distance_loss,
        "dice_coef": dice_coef
      }
    )
    
  else:
    with tf.device("/device:GPU:0"):
      model = unet_model(img_size)
      adam_opt = keras.optimizers.Adam(learning_rate = 2e-4)
      model.compile(optimizer = adam_opt, loss = jaccard_distance_loss, metrics = [dice_coef])
    

      fit = model.fit(X_train, Y_train, 
      batch_size=6, 
      epochs = 80,
      validation_data = (X_val, Y_val),
      callbacks = [
        checkpointer,
        reduce_learning_rate
      ]
    )


  print(model.summary())

if __name__ == "__main__":
    main()