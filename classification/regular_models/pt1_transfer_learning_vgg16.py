#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage
import skimage
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte
from pt0_load_data import AugmentationSequence 
import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma, CLAHE
)

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import telebot
import time
import tensorflow as tf
from tensorflow import keras
import datetime
import lime
from lime import lime_image

import pandas as pd
import pickle

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
    
SEED = 587
TELEBOT_TOKEN = "2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA"
np.random.seed(SEED)

img_size = 300
bot = telebot.TeleBot("2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA")
bot.config['api_key'] = TELEBOT_TOKEN
bot.get_me()


### Create Model

def create_model():
  
  inputs = keras.Input(shape = (img_size, img_size, 3))
  
  base_model = keras.applications.VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (img_size, img_size, 3)
  )
  base_model.trainable = False
  
  x = base_model(inputs, training = False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(1024, activation = "relu")(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.5)(x)
  x = keras.layers.Dense(1024, activation = "relu")(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.5)(x)
  x = keras.layers.Dense(512, activation = "relu")(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.5)(x)
  output = keras.layers.Dense(3, activation = 'softmax')(x)

  model = keras.Model(inputs = inputs, outputs = output)

  return model, base_model


def main():
  np.random.seed(SEED)
  ### Load Datasets

  file_pi = open('input/train_dataset.pkl', 'rb') 
  train_generator =  pickle.load(file_pi)

  X_val = np.load('input/X_val.npy')
  Y_val = np.load('input/Y_val.npy')

  with open('input/epochs.txt', 'r') as file:
      steps_per_epoch = file.read().rstrip()

  steps_per_epoch = int(steps_per_epoch)

  print(f"Number of steps {steps_per_epoch}")

  # ## Fit Model

  reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
    monitor = "loss", 
    factor = 0.5, 
    patience = 3, 
    verbose = 1
  )

  model_name = "cache/tl_vgg16_cd.h5"
  if (os.path.exists(model_name)):
    os.remove(model_name)

  checkpointer = keras.callbacks.ModelCheckpoint(
    model_name,
    monitor = "val_accuracy",
    verbose = 1, 
    save_best_only = True
  )

  if (os.path.exists(model_name)):
    model = keras.models.load_model(model_name)
    print("existe")
    
  else:
    model, base_model = create_model()
    adam_opt = keras.optimizers.Adam(learning_rate = 0.001 )
    model.compile(optimizer = adam_opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
    start = time.time()    
    fit = model.fit(train_generator, 
      steps_per_epoch = steps_per_epoch, 
      epochs = 50,
      validation_data = (X_val, Y_val),
      callbacks = [
        checkpointer,
        reduce_learning_rate
      ]
    )
    end = time.time()

  print("Modelo Base")
  print(model.summary())

  final_train1 = end-start

  time_hours = str(datetime.timedelta(seconds=final_train1))
  bot.send_message("-600800507", f'Rede {model_name} - Treinamento Finalizado em {time_hours}')

  ## Inicialização do finetune e salvando ele
  model_name = "cache/tl_vgg16_finetune_cd.h5"
  if (os.path.exists(model_name)):
    os.remove(model_name)

  if (os.path.exists(model_name)):
    print("existe")
    model = keras.models.load_model(model_name)
    
  else:
    base_model.trainable = True
    adam_opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = adam_opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.save('cache/tl_vgg16_finetune_cd.h5')
    
  #### Save times
  text_file = open("input/time_train.txt", "wt")
  n = text_file.write(str(final_train1))
  text_file.close()


if __name__ == "__main__":
    main()