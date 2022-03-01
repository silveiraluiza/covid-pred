#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pt0_load_data import AugmentationSequence
import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, ElasticTransform,
    RandomBrightness, RandomContrast, RandomGamma, CLAHE
)
import datetime
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import telebot
import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle
import sys
import argparse

parser = argparse.ArgumentParser(description='Processo treinamento de modelo')
parser.add_argument('-m','--model',  type=str , help='modelo', required=True)

args = parser.parse_args()

model_name = args.model

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
    
    
TELEBOT_TOKEN = "2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA"

bot = telebot.TeleBot("2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA")
bot.config['api_key'] = TELEBOT_TOKEN
bot.get_me()

SEED = 587

def main():
  np.random.seed(SEED)
  tf.keras.backend.clear_session()

  ### Load Datasets
  file_pi = open('input/train_dataset.pkl', 'rb') 
  train_generator =  pickle.load(file_pi)

  X_val = np.load('input/X_val.npy')
  Y_val = np.load('input/Y_val.npy')

  with open('input/epochs.txt', 'r') as file:
      steps_per_epoch = file.read().rstrip()

  steps_per_epoch = int(steps_per_epoch)

  print(f"Number of steps {steps_per_epoch}")

  if model_name == "VGG16":
    model_name = "cache/tl_vgg16_finetune.h5"
  
  elif model_name == "DenseNet":
    model_name = "cache/tl_densenet121_finetune.h5"

  elif model_name == "InceptionResNet":
    model_name = "cache/tl_inceptionresnet_finetune.h5"

  elif model_name == "ResNet152V2":
    model_name = "cache/tl_resnet152_finetune.h5"

  ### Model Finetune
 
  model = keras.models.load_model(model_name)

  print("Modelo Finetune")
  print(model.summary())

  checkpointer = keras.callbacks.ModelCheckpoint(
    model_name,
    monitor = "val_accuracy",
    verbose = 1, 
    save_best_only = True
  )

  reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
    monitor = "loss", 
    factor = 0.5, 
    patience = 3, 
    verbose = 1
  )

  start = time.time()    
  fit = model.fit(train_generator, 
      steps_per_epoch = steps_per_epoch, 
      epochs = 100,
      validation_data = (X_val, Y_val),
      callbacks = [
        checkpointer,
        reduce_learning_rate
      ]
    )

  end = time.time()

  final_train1 = end-start

  time_hours = str(datetime.timedelta(seconds=final_train1))
  bot.send_message("-600800507", f'Rede {model_name} - Treinamento Finetune Finalizado em {time_hours}')

  #### Save Time

  text_file = open("input/time_train_2.txt", "wt")
  n = text_file.write(str(final_train1))
  text_file.close()


if __name__ == "__main__":
  main()
