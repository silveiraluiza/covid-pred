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

import pickle
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

import lime
from lime import lime_image

import pandas as pd

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

np.random.seed(587)

# Função para carregar as imagens

def load_images(root_folder, train_test, img_size):

  images = []
  labels = []

  for idx, pathogen in enumerate(["Opacity", "COVID-19", "Normal"]):
    for img_filename in os.listdir(os.path.join(root_folder, train_test, pathogen)):
      img = cv2.imread(os.path.join(root_folder, train_test, pathogen, img_filename), 0)
      img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
      img = skimage.img_as_float32(img)
      images.append(img)
      labels.append(idx)

  return images, labels

# Classe Augmentation

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
    for idx in range(batch_x.shape[0]):
      aug = self.augment(image = batch_x[idx,:,:])
      aug_x[idx,:,:] = aug["image"]

    return np.stack((aug_x,) * 3, axis = -1), batch_y


def main():

  ### Load Images

  root_folder = "../../3_Images"

  img_size = 300

  images_train, labels_train = load_images(root_folder, "train", img_size)
  images_test, labels_test = load_images(root_folder, "test", img_size)

  print("Tamanhos train e test")
  print(len(images_train))
  print(len(images_test))


  ### Cria Val, Train e Test

  X_train = np.array(images_train).reshape((len(images_train), img_size, img_size))
  Y_train = keras.utils.to_categorical(labels_train)
  X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state = 587)

  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 587)
  X_val = np.stack((X_val,) * 3, axis = -1)

  X_test = np.array(images_test).reshape((len(images_test), img_size, img_size))
  X_test = np.stack((X_test,) * 3, axis = -1)
  Y_test = keras.utils.to_categorical(labels_test)
  X_test, Y_test = sklearn.utils.shuffle(X_test, Y_test)

  print(np.unique(np.argmax(Y_train, axis = 1), return_counts = True))
  print(np.unique(np.argmax(Y_val, axis = 1), return_counts = True))
  print(np.unique(np.argmax(Y_test, axis = 1), return_counts = True))


  # ## Augmentation

  augment = Compose([
    HorizontalFlip(),
    ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 15, border_mode = cv2.BORDER_CONSTANT),
    ElasticTransform(sigma = 20, alpha_affine = 20, border_mode = cv2.BORDER_CONSTANT),
    RandomBrightness(),
    RandomContrast(),
    RandomGamma()
  ])

  batch_size = 16
  train_generator = AugmentationSequence(X_train, Y_train, batch_size, augment)
  steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)


  ### Salvando datasets

  with open('input/train_dataset.pkl', 'wb') as file:
      pickle.dump(train_generator, file)

  np.save('input/X_test.npy', X_test)
  np.save('input/Y_test.npy', Y_test)
  np.save('input/X_val.npy', X_val)
  np.save('input/Y_val.npy', Y_val)
  np.save('input/X_train.npy', X_train)
  np.save('input/Y_train.npy', Y_train)

  del(X_test,X_train,Y_test,Y_train,X_val,Y_val)

  ### Agora o Test2

  root_folder = "../../3_Images_Tests"

  img_size = 300

  images_train, labels_train = load_images(root_folder, "train", img_size)
  images_test, labels_test = load_images(root_folder, "test", img_size)

  print("Tamanho train e test 2")
  print(len(images_train))
  print(len(images_test))

  # Formatando

  X_train = np.array(images_train).reshape((len(images_train), img_size, img_size))
  Y_train = keras.utils.to_categorical(labels_train)
  X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train, random_state = 587)

  X_test = np.array(images_test).reshape((len(images_test), img_size, img_size))
  X_test = np.stack((X_test,) * 3, axis = -1)
  Y_test = keras.utils.to_categorical(labels_test)
  X_test, Y_test = sklearn.utils.shuffle(X_test, Y_test)


  ## Salvando datasets 2

  np.save('input/X_test2.npy', X_test)
  np.save('input/Y_test2.npy', Y_test)
  np.save('input/X_train2.npy', X_train)
  np.save('input/Y_train2.npy', Y_train)

  # Salvando epochs
  text_file = open("input/epochs.txt", "wt")
  n = text_file.write(str(steps_per_epoch))
  text_file.close()

  print("Criação das bases finalizada")
  bot.send_message("-600800507", f'Criação das bases finalizada')

if __name__ == "__main__":
    main()