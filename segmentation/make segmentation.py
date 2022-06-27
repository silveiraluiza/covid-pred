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
from skimage import img_as_float32, img_as_ubyte, img_as_float64
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

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    print(gpus)
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
    
img_size = 400


# ## Functions

# In[2]:


reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
  monitor = "loss", 
  factor = 0.5, 
  patience = 3, 
  verbose = 1
)

checkpointer = keras.callbacks.ModelCheckpoint(
  "unet.h5", 
  verbose = 1, 
  save_best_only = True
)

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


# ## Load Unet

# In[3]:


if (os.path.exists("unet.h5")):
  model = keras.models.load_model("unet.h5",
    custom_objects = {
      "jaccard_distance_loss": jaccard_distance_loss,
      "dice_coef": dice_coef
    }
  )
  


# ## Metrics

# In[4]:


X_val = np.load('input/X_val.npy')
Y_val = np.load('input/Y_val.npy')


with open('input/epochs.txt', 'r') as file:
    steps_per_epoch = file.read().rstrip()

steps_per_epoch = int(steps_per_epoch)


# In[5]:


X_val[0].shape


# In[6]:


iou_val, dice_val = model.evaluate(X_val, Y_val, verbose = False)


# In[ ]:


X_train = np.load('input/X_train.npy')
Y_train = np.load('input/Y_train.npy')


iou_train, dice_train = model.evaluate(X_train, Y_train, verbose = False)


# In[ ]:


X_test = np.load('input/X_test.npy')
Y_test = np.load('input/Y_test.npy')


iou_test, dice_test = model.evaluate(X_test, Y_test, verbose = False)


# In[ ]:


print("Jaccard distance (IoU) train: %f" % iou_train)
print("Dice coeffient train: %f" % dice_train)
print("Jaccard distance (IoU) validation: %f" % iou_val)
print("Dice coeffient validation: %f" % dice_val)
print("Jaccard distance (IoU) test: %f" % iou_test)
print("Dice coeffient test: %f" % dice_test)


# In[ ]:


nimages = X_train.shape[0]
iou_train = []
dice_train = []
for idx in range(nimages):
  iou, dice = model.evaluate(X_train[idx:idx+1,:,:], Y_train[idx:idx+1,:,:], verbose = False)
  iou_train.append(iou)
  dice_train.append(dice)

print("Jaccard distance (IoU) train: %f (+-%f)" % (np.mean(iou_train), np.std(iou_train)))
print("Dice coeffient train: %f (+-%f)" % (np.mean(dice_train), np.std(dice_train)))

nimages = X_val.shape[0]
iou_val = []
dice_val = []
for idx in range(nimages):
  iou, dice = model.evaluate(X_val[idx:idx+1,:,:], Y_val[idx:idx+1,:,:], verbose = False)
  iou_val.append(iou)
  dice_val.append(dice)

print("Jaccard distance (IoU) validation: %f (+-%f)" % (np.mean(iou_val), np.std(iou_val)))
print("Dice coeffient validation: %f (+-%f)" % (np.mean(dice_val), np.std(dice_val)))


nimages = X_test.shape[0]
iou_test = []
dice_test = []
for idx in range(nimages):
  iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
  iou_test.append(iou)
  dice_test.append(dice)

print("Jaccard distance (IoU) test: %f (+-%f)" % (np.mean(iou_test), np.std(iou_test)))
print("Dice coeffient test: %f (+-%f)" % (np.mean(dice_test), np.std(dice_test)))


# In[ ]:


shenzhen_test_ids = []
jsrt_test_ids = []
montgomery_test_ids = []
v7labs_test_ids = []
other_test_ids = []

count = 0
datasets_ids = [shenzhen_test_ids, jsrt_test_ids, montgomery_test_ids, v7labs_test_ids, other_test_ids]
with open("input/test_ids.txt") as fp:
    for line in fp:
        new_list = line.strip()
        datasets_ids[count].append(new_list)
        count += 1


# In[ ]:


import ast

shenzhen_test_ids = ast.literal_eval(shenzhen_test_ids[0])
jsrt_test_ids = ast.literal_eval(jsrt_test_ids[0])
montgomery_test_ids = ast.literal_eval(montgomery_test_ids[0])
v7labs_test_ids = ast.literal_eval(v7labs_test_ids[0])
other_test_ids = ast.literal_eval(other_test_ids[0])


# In[ ]:


nimages = X_test.shape[0]
iou_test = []
dice_test = []
with tf.device("/gpu:1"):
  for idx in range(nimages):
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_test.append(iou)
    dice_test.append(dice)

print("Jaccard distance (IoU) test: %f (+-%f)" % (np.mean(iou_test), np.std(iou_test)))
print("Dice coeffient test: %f (+-%f)" % (np.mean(dice_test), np.std(dice_test)))


iou_shenzhen = []
dice_shenzhen = []
with tf.device("/gpu:1"):
  for idx in shenzhen_test_ids:
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_shenzhen.append(iou)
    dice_shenzhen.append(dice)

print("Jaccard distance (IoU) Shenzhen: %f (+-%f)" % (np.mean(iou_shenzhen), np.std(iou_shenzhen)))
print("Dice coeffient Shenzhen: %f (+-%f)" % (np.mean(dice_shenzhen), np.std(dice_shenzhen)))


iou_montgomery = []
dice_montgomery = []
with tf.device("/gpu:1"):
  for idx in montgomery_test_ids:
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_montgomery.append(iou)
    dice_montgomery.append(dice)

print("Jaccard distance (IoU) Montgomery: %f (+-%f)" % (np.mean(iou_montgomery), np.std(iou_montgomery)))
print("Dice coeffient Montgomery: %f (+-%f)" % (np.mean(dice_montgomery), np.std(dice_montgomery)))


iou_jsrt = []
dice_jsrt = []
with tf.device("/gpu:1"):
  for idx in jsrt_test_ids:
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_jsrt.append(iou)
    dice_jsrt.append(dice)

print("Jaccard distance (IoU) JSRT: %f (+-%f)" % (np.mean(iou_jsrt), np.std(iou_jsrt)))
print("Dice coeffient JSRT: %f (+-%f)" % (np.mean(dice_jsrt), np.std(dice_jsrt)))


iou_v7labs = []
dice_v7labs = []
with tf.device("/gpu:1"):
  for idx in v7labs_test_ids:
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_v7labs.append(iou)
    dice_v7labs.append(dice)

print("Jaccard distance (IoU) v7labs: %f (+-%f)" % (np.mean(iou_v7labs), np.std(iou_v7labs)))
print("Dice coeffient v7labs: %f (+-%f)" % (np.mean(dice_v7labs), np.std(dice_v7labs)))



iou_manual = []
dice_manual = []
with tf.device("/gpu:1"):
  for idx in other_test_ids:
    iou, dice = model.evaluate(X_test[idx:idx+1,:,:], Y_test[idx:idx+1,:,:], verbose = False)
    iou_manual.append(iou)
    dice_manual.append(dice)

print("Jaccard distance (IoU) manual: %f (+-%f)" % (np.mean(iou_manual), np.std(iou_manual)))
print("Dice coeffient manual: %f (+-%f)" % (np.mean(dice_manual), np.std(dice_manual)))


# ## Visualizando a segmentação

# In[ ]:


idx = 26

test_img = X_test[idx,:,:,:].reshape((1, img_size, img_size, 1))
test_mask = Y_test[idx,:,:,:].reshape((1, img_size, img_size, 1))
pred_mask = model.predict(test_img)
pred_mask = np.uint8(pred_mask > 0.5)
post_pred_mask = skimage.morphology.erosion(pred_mask[0,:,:,0], skimage.morphology.square(5))
post_pred_mask = skimage.morphology.dilation(post_pred_mask, skimage.morphology.square(10))


f = plt.figure(figsize = (20, 10))
f.add_subplot(1, 4, 1)
plt.imshow(img_as_float64(test_img[0,:,:,0]), cmap = "gray")
f.add_subplot(1, 4, 2)
plt.imshow(test_mask[0,:,:,0], cmap = "gray")
f.add_subplot(1, 4, 3)
plt.imshow(pred_mask[0,:,:,0], cmap = "gray")
f.add_subplot(1, 4, 4)
plt.imshow(post_pred_mask, cmap = "gray")


# In[ ]:


def crop_image(img, mask):
  crop_mask = mask > 0
  m, n = mask.shape
  crop_mask0, crop_mask1 = crop_mask.any(0), crop_mask.any(1)
  col_start, col_end = crop_mask0.argmax(), n - crop_mask0[::-1].argmax()
  row_start, row_end = crop_mask1.argmax(), m - crop_mask1[::-1].argmax()
  return img[row_start:row_end, col_start:col_end], mask[row_start:row_end, col_start:col_end]
  
#idx = 70
idx = 2
test_img = X_test[idx,:,:,:].reshape((1, img_size, img_size, 1))
test_mask = Y_test[idx,:,:,:].reshape((1, img_size, img_size, 1))
pred_mask = model.predict(test_img)[0,:,:,0]
pred_mask = np.uint8(pred_mask > 0.5)
open_pred_mask = skimage.morphology.erosion(pred_mask, skimage.morphology.square(5))
open_pred_mask = skimage.morphology.dilation(open_pred_mask, skimage.morphology.square(5))
post_pred_mask = skimage.morphology.dilation(open_pred_mask, skimage.morphology.square(5))

crop_img, crop_mask = crop_image(test_img[0,:,:,0], pred_mask)

crop_img_masked = crop_img * crop_mask

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(img_as_float64(test_img[0,:,:,0]), cmap = "gray")
f.add_subplot(2, 2, 2)
plt.imshow(post_pred_mask, cmap = "gray")
f.add_subplot(2, 2, 3)
plt.imshow(img_as_float64(crop_img), cmap = "gray")
f.add_subplot(2, 2, 4)
plt.imshow(crop_mask, cmap = "gray")


# In[ ]:


f = plt.figure(figsize = (20, 20))
f.add_subplot(1, 1, 1)
plt.imshow(img_as_float32(test_img[0,:,:,0]), cmap = "gray")
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])


# In[ ]:


f = plt.figure(figsize = (20, 20))
f.add_subplot(1, 1, 1)
plt.imshow(open_pred_mask, cmap = "gray")
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])


# In[ ]:


f = plt.figure(figsize = (20, 20))
f.add_subplot(1, 1, 1)
plt.imshow(post_pred_mask, cmap = "gray")
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])


# In[ ]:


f = plt.figure(figsize = (20, 20))
f.add_subplot(1, 1, 1)
plt.imshow(crop_img_masked, cmap = "gray")
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

'''
# # Make Predictions

# In[ ]:


source_folders = [ "Cohen", "RSNA", "Actualmed", "Figure1", "KaggleCRD", "RICORD", 'BIMCV', 'OCT'] #"BIMCV"
pneumonia_folders = ["Bacteria", "Fungi", "Virus", "Pneumonia", "Lung Opacity"]
pathogen_folders = ["Bacteria", "Fungi", "Virus", "Pneumonia", "Lung Opacity", "COVID-19", "Normal"]

dest_folder = "../2_Raw_Seg/"
root_folder = "../2_Raw/"
#masks_folder = os.path.join(root_folder, "Masks")

img_size = 400
dim = (img_size,img_size)

if os.path.isdir(dest_folder):
  shutil.rmtree(dest_folder)

if not os.path.isdir(dest_folder):
  os.makedirs(dest_folder)
  

for path, subdirs, files in os.walk(root_folder):
    for dirs in subdirs:
        path = path.replace(root_folder, dest_folder)
        dir_path = os.path.join(path, dirs)
        print(dir_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

for path, subdirs, files in os.walk(root_folder):
    for name in files:
        img_path = os.path.join(path, name)
        
        print(img_path)
        
        if len(img_path.split("/")) == 6:
            ref, source, data, patgn, pat2, name_im = re.split("/", img_path)
            img_filename = os.path.join(dest_folder,data,patgn,pat2,name_im)
        elif len(img_path.split("/")) == 7:
            ref, source, data, patgn, pat2, pat3, name_im = re.split("/", img_path)
            img_filename = os.path.join(dest_folder,data,patgn,pat2,pat3,name_im)
        else:
            ref, source, data, patgn, name_im = re.split("/", img_path)
            img_filename = os.path.join(dest_folder,data,patgn,name_im)
            
        mask_dir = "Masks"
        mask_filename = os.path.join(dest_folder,data,mask_dir,name_im)
     
        if not mask_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            mask_filename = mask_filename + ".png"
            
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_filename = img_filename + ".png"
            
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        print(img.shape)
        img = img_as_float32(img).reshape((1, img_size, img_size, 1))
    
        mask_pred = model.predict(img)[0,:,:,0]
        mask_pred = np.uint8(mask_pred > 0.5)
        mask_pred = skimage.morphology.erosion(mask_pred, skimage.morphology.square(5))
        mask_pred = skimage.morphology.dilation(mask_pred, skimage.morphology.square(10))
        
        img = img_as_ubyte(img[0,:,:,0])

        crop_img, crop_mask = crop_image(img, mask_pred)
        wdt, hgt = crop_img.shape
        
        if wdt < 200 or hgt < 200:
            print(img_filename)
            print("too small")
          #continue

        crop_img = crop_img * crop_mask

        crop_img = cv2.resize(crop_img, (300, 300), interpolation = cv2.INTER_CUBIC)
        crop_mask = cv2.resize(crop_mask, (300, 300), interpolation = cv2.INTER_CUBIC)

        imsave(os.path.join(mask_filename), crop_mask * 255)
        imsave(os.path.join(img_filename), crop_img)
        print('-----------------------')

'''
# ## Predict GAN

# In[ ]:


source_folders = [ "COVID-19_Imgs", "Normal_Imgs", "Pneumonia_Imgs"]

dest_folder = "../augmentation/Results/Seg/"
root_folder = "../augmentation/Results/"
#masks_folder = os.path.join(root_folder, "Masks")

img_size = 400
dim = (img_size,img_size)

if os.path.isdir(dest_folder):
  shutil.rmtree(dest_folder)

if not os.path.isdir(dest_folder):
  os.makedirs(dest_folder)
  

for source in source_folders:
    new_dest = os.path.join(dest_folder, source)
    os.makedirs(new_dest)
    for path, subdirs, files in os.walk(os.path.join(root_folder, source)):
        for name in files:
            img_path = os.path.join(path, name)

            print(img_path)

            ref, source, data, patgn, name_im = re.split("/", img_path)
            img_filename = img_path


            mask_filename = os.path.join(dest_folder,patgn,name_im)
            
            if not mask_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                mask_filename = mask_filename + ".png"

            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_filename = img_filename + ".png"

            img = imread(img_path, cv2.IMREAD_GRAYSCALE)

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            print(img.shape)
            img = img_as_float32(img).reshape((1, img_size, img_size, 1))

            mask_pred = model.predict(img)[0,:,:,0]
            mask_pred = np.uint8(mask_pred > 0.5)
            mask_pred = skimage.morphology.erosion(mask_pred, skimage.morphology.square(5))
            mask_pred = skimage.morphology.dilation(mask_pred, skimage.morphology.square(10))

            img = img_as_ubyte(img[0,:,:,0])

            crop_img, crop_mask = crop_image(img, mask_pred)
            wdt, hgt = crop_img.shape

            if wdt < 200 or hgt < 200:
                print(img_filename)
                print("too small")
              #continue

            crop_img = crop_img * crop_mask

            crop_img = cv2.resize(crop_img, (300, 300), interpolation = cv2.INTER_CUBIC)
            crop_mask = cv2.resize(crop_mask, (300, 300), interpolation = cv2.INTER_CUBIC)

            imsave(os.path.join(mask_filename), crop_mask * 255)
            imsave(os.path.join(img_filename), crop_img)
            print('-----------------------')

