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
    RandomBrightness, RandomContrast, RandomGamma, GaussNoise, GridDistortion
)


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices("GPU")


from glob import glob
from tqdm import tqdm

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

def main():

  X_train = np.load('input/X_train.npy')
  Y_train = np.load('input/Y_train.npy')

  augment = Compose([
    HorizontalFlip(),
    ShiftScaleRotate(rotate_limit = 45, border_mode = cv2.BORDER_CONSTANT),
    ElasticTransform(border_mode = cv2.BORDER_CONSTANT),
    GridDistortion(border_mode = cv2.BORDER_CONSTANT),
    RandomBrightness(),
    RandomContrast(),
    RandomGamma()
  ])
  print("Inicia Augmentation")
  batch_size = 16
  train_generator = AugmentationSequence(X_train, Y_train, batch_size, augment)
  steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

  with open('input/train_dataset.pkl', 'wb') as file:
      pickle.dump(train_generator, file)



  print(X_train.shape)



  # Create aug_dataset
  X_aug = []
  Y_aug = []
  print("Transformando em Array")
  for batch in range(train_generator.__len__()):
      x_aug, y_aug = train_generator.__getitem__(batch)
      for img, mask in zip(x_aug,y_aug):
          X_aug.append(img)
          Y_aug.append(mask)
          
  X_aug = np.array(X_aug)
  Y_aug = np.array(Y_aug)


 
  print(X_aug.shape)


  del(x_aug,y_aug,train_generator)

  print("Juntando ao X_train")

  X_train =  np.concatenate([X_train, X_aug], axis=0)
  Y_train = np.concatenate([Y_train, Y_aug], axis=0)

  print(X_train.shape)


  ### Save Dataset

  np.save('input/X_train.npy', X_train)
  np.save('input/Y_train.npy', Y_train)
  
  # Salvando epochs
  text_file = open("input/epochs.txt", "wt")
  n = text_file.write(str(steps_per_epoch))
  text_file.close()
  print("Salvo e Finalizado")

if __name__ == "__main__":
    main()

