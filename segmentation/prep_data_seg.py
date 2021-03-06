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
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


from glob import glob
from tqdm import tqdm

DILATE_KERNEL = np.ones((15, 15), np.uint8)
MONTGOMERY_LEFT_MASK_DIR = "extra data/Montgomery/ManualMask/leftMask/"
MONTGOMERY_IMAGE_DIR = "extra data/Montgomery/images/"
MONTGOMERY_RIGHT_MASK_DIR = "extra data/Montgomery/ManualMask/rightMask/"
SEGMENTATION_DIR = "extra data/Montgomery/ManualMask/joined/"
SEGMENTATION_DILATE_DIR = "extra data/Montgomery/ManualMask/dilate/"




    
def create_montgomery_masks():
  if os.path.exists(SEGMENTATION_DIR):
    shutil.rmtree(SEGMENTATION_DIR)
    shutil.rmtree(SEGMENTATION_DILATE_DIR)

  os.mkdir(SEGMENTATION_DIR)
  os.mkdir(SEGMENTATION_DILATE_DIR)

  montgomery_left_mask_dir = glob(os.path.join(MONTGOMERY_LEFT_MASK_DIR, '*.png'))
  montgomery_test = montgomery_left_mask_dir

  for left_image_file in tqdm(montgomery_left_mask_dir):
      base_file = os.path.basename(left_image_file)
      image_file = os.path.join(MONTGOMERY_IMAGE_DIR, base_file)
      right_image_file = os.path.join(MONTGOMERY_RIGHT_MASK_DIR, base_file)

      image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
      left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
      right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)
      
      image = cv2.resize(image, (400, 400))
      left_mask = cv2.resize(left_mask, (400, 400))
      right_mask = cv2.resize(right_mask, (400, 400))
      
      mask = np.maximum(left_mask, right_mask)
      mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
      

      cv2.imwrite(os.path.join(SEGMENTATION_DIR, base_file), mask)
      cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), mask_dilate)


def create_cohen_data():
  ### Cohen
  root_folder = "../3_Images"
  masks_folder = os.path.join(root_folder, "masks")

  img_size = 400



  images = []
  labels = []
  manual_images = []
  v7labs_images = []

  for mask_path in os.listdir(masks_folder):
    mask = imread(os.path.join(masks_folder, mask_path), cv2.IMREAD_GRAYSCALE)
    print('Original Dimensions : ',mask.shape)
    if len(mask.shape) == 3:
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.float32(mask / 255.0)

    width = int(400)
    height = int(400)
    dim = (width, height)

    # resize image
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    labels.append(mask)

    target, source, pathogen, pid, offset, _ = re.split("[_.]", mask_path)
    img_path = "%s_%s_%s_%s.png" % (source, pathogen, pid, offset)
    img = imread(os.path.join(root_folder, target, pathogen, img_path), cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('Original Dimensions : ',img.shape)

    img = img_as_float32(img)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    if source == "Cohen":
      v7labs_images.append(img)
    else:
      manual_images.append(img)
    
    images.append(img)
    
  print(len(labels))
  return images, labels, v7labs_images


'''
# Show some of our own CXR and masks
f = plt.figure()
f.add_subplot(3, 2, 1)
plt.imshow(images[50], cmap = "gray")
f.add_subplot(3, 2, 2)
plt.imshow(labels[50], cmap = "gray")
f.add_subplot(3, 2, 3)
plt.imshow(images[20], cmap = "gray")
f.add_subplot(3, 2, 4)
plt.imshow(labels[20], cmap = "gray")
f.add_subplot(3, 2, 5)
plt.imshow(images[30], cmap = "gray")
f.add_subplot(3, 2, 6)
plt.imshow(labels[30], cmap = "gray")

'''
def create_montgomery_data(images, labels):
  # ## Montgomery

  # Also load the Montgomery dataset that already contains manually segmented masks
  montgomery_images = []
  montgomery_labels = []
  montgomery_folder = "extra data/Montgomery/"
  for img_path in os.listdir(os.path.join(montgomery_folder, "images")):
      print(img_path)
      img_path_new = os.path.join(montgomery_folder, "images", img_path)
      img = imread(img_path_new, cv2.IMREAD_GRAYSCALE)
      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
      print('Original Dimensions : ',img.shape)
  
      width = int(400)
      height = int(400)
      dim = (width, height)

      # resize image
      img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

      print('Resized Dimensions : ',img.shape)
  
      
      
      img = img_as_float32(img)
      montgomery_images.append(img)
      images.append(img)

      img_path_mask = os.path.join(montgomery_folder, "ManualMask", "joined", img_path)
      mask = imread(img_path_mask, cv2.IMREAD_GRAYSCALE)
      if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
      
      mask = np.float32(mask / 255.0)
      print('Original Dimensions : ',mask.shape)
      montgomery_labels.append(mask)
      labels.append(mask)


  print(len(montgomery_labels))
  return images, labels, montgomery_images

'''
f = plt.figure()
f.add_subplot(3, 2, 1)
plt.imshow(montgomery_images[10], cmap = "gray")
f.add_subplot(3, 2, 2)
plt.imshow(montgomery_labels[10], cmap = "gray")
f.add_subplot(3, 2, 3)
plt.imshow(montgomery_images[20], cmap = "gray")
f.add_subplot(3, 2, 4)
plt.imshow(montgomery_labels[20], cmap = "gray")
f.add_subplot(3, 2, 5)
plt.imshow(montgomery_images[30], cmap = "gray")
f.add_subplot(3, 2, 6)
plt.imshow(montgomery_labels[30], cmap = "gray")
'''

def create_jsrt_data(images, labels):
  # ## JSRT

  # Also load the JSRT dataset that already contains manually segmented masks
  jsrt_images = []
  jsrt_labels = []
  jsrt_folder = "extra data/JSRT/"

  for img_path in os.listdir(os.path.join(jsrt_folder, "Images")):
      img = imread(os.path.join(jsrt_folder, "Images", img_path), cv2.IMREAD_GRAYSCALE)
      img = img_as_float32(img)
      if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
      print('Original Dimensions : ',img.shape)

      width = int(400)
      height = int(400)
      dim = (width, height)

      # resize image
      img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

      print('Resized Dimensions : ',img.shape)
  
        
      jsrt_images.append(img)
      images.append(img)
    
      mask = imread(os.path.join(jsrt_folder, "Masks", img_path), cv2.IMREAD_GRAYSCALE)
      if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

      mask = np.float32(mask / 255.0)
      
      print('Original Dimensions : ',mask.shape)
      jsrt_labels.append(mask)
      labels.append(mask)

  print(len(jsrt_labels))
  return images,labels, jsrt_images

'''

f = plt.figure()
f.add_subplot(3, 2, 1)
plt.imshow(jsrt_images[10], cmap = "gray")
f.add_subplot(3, 2, 2)
plt.imshow(jsrt_labels[10], cmap = "gray")
f.add_subplot(3, 2, 3)
plt.imshow(jsrt_images[20], cmap = "gray")
f.add_subplot(3, 2, 4)
plt.imshow(jsrt_labels[20], cmap = "gray")
f.add_subplot(3, 2, 5)
plt.imshow(jsrt_images[30], cmap = "gray")
f.add_subplot(3, 2, 6)
plt.imshow(jsrt_labels[30], cmap = "gray")

'''

def create_shenzen_data(images,labels):
# ## Shenzen

  # Also load the Shenzhen dataset that already contains manually segmented masks
  shenzhen_images = []
  shenzhen_labels = []
  shenzhen_folder = "extra data/Shenzen/"
  for img_path in os.listdir(os.path.join(shenzhen_folder, "images")):
    img = imread(os.path.join(shenzhen_folder, "images", img_path), cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_name = img_path.split(".")[0] + "_mask.png"
    mask_path = os.path.join(shenzhen_folder, "Mask", mask_name)

    if os.path.exists(mask_path):  
        print('Original Dimensions : ',img.shape)

        width = int(400)
        height = int(400)
        dim = (width, height)

        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        print('Resized Dimensions : ',img.shape)  
        img = img_as_float32(img)
        shenzhen_images.append(img)
        images.append(img)

        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print('Original Dimensions : ',mask.shape)
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        mask = np.float32(mask / 255.0)
        print('Resized Dimensions : ',mask.shape)  
        shenzhen_labels.append(mask)
        labels.append(mask)

  print(len(shenzhen_labels))
  return images,labels,shenzhen_images

'''
f = plt.figure()
f.add_subplot(3, 2, 1)
plt.imshow(shenzhen_images[10], cmap = "gray")
f.add_subplot(3, 2, 2)
plt.imshow(shenzhen_labels[10], cmap = "gray")
f.add_subplot(3, 2, 3)
plt.imshow(shenzhen_images[20], cmap = "gray")
f.add_subplot(3, 2, 4)
plt.imshow(shenzhen_labels[20], cmap = "gray")
f.add_subplot(3, 2, 5)
plt.imshow(shenzhen_images[30], cmap = "gray")
f.add_subplot(3, 2, 6)
plt.imshow(shenzhen_labels[30], cmap = "gray")

'''

def main():
  img_size = 400
  root_folder = "../3_Images"
  masks_folder = os.path.join(root_folder, "masks")

  images, labels, v7labs_images = create_cohen_data()
  images, labels, montgomery_images = create_montgomery_data(images, labels)
  images, labels, jsrt_images = create_jsrt_data(images, labels)
  images, labels, shenzhen_images = create_shenzen_data(images, labels)

  X = np.array(images).reshape((len(images), img_size, img_size, 1))
  Y = np.array(labels).reshape((len(labels), img_size, img_size, 1))
  X, Y = shuffle(X, Y, random_state = 5564)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 5564)
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, random_state = 5564)

  print(X_train.shape)
  print(X_val.shape)
  print(X_test.shape)
  print(X.shape)



  shenzhen_test_ids = []
  jsrt_test_ids = []
  montgomery_test_ids = []
  v7labs_test_ids = []
  other_test_ids = []

  nimages = X_test.shape[0]
  for idx in range(nimages):
    test_image = X_test[idx,:,:,0]
    if any(np.array_equal(test_image, x) for x in shenzhen_images):
      shenzhen_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in montgomery_images):
      montgomery_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in jsrt_images):
      jsrt_test_ids.append(idx)
    elif any(np.array_equal(test_image, x) for x in v7labs_images):
      v7labs_test_ids.append(idx)
    else:
      other_test_ids.append(idx)
      
  del(images,labels,shenzhen_images,montgomery_images,jsrt_images,v7labs_images)
      
  lines = [shenzhen_test_ids, jsrt_test_ids, montgomery_test_ids, v7labs_test_ids, other_test_ids]
  with open('input/test_ids.txt', 'w') as f:
      for line in lines:
          f.write(str(line))
          f.write('\n')

  # ## Save Dataset
  np.save('input/X_test.npy', X_test)
  np.save('input/Y_test.npy', Y_test)
  np.save('input/X_val.npy', X_val)
  np.save('input/Y_val.npy', Y_val)
  np.save('input/X_train.npy', X_train)
  np.save('input/Y_train.npy', Y_train)






if __name__ == "__main__":
    main()

