import os
import shutil
import re
import random
import cv2
import numpy as np
import telebot
# Split data in training/testing using a 80/20 distribution
# The distribution is centered around the patient since each person can have multiple images in different days
# So the idea is to keep all images from the same patient all in the same set (either train or test)
TELEBOT_TOKEN = "2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA"

bot = telebot.TeleBot("2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA")
bot.config['api_key'] = TELEBOT_TOKEN
bot.get_me()

# The dataset is composed by CXR images of pneumonia (any other, except COVID-19), COVID-19 and Normal.
source_folders = [ "Cohen", "RSNA", "Actualmed", "Figure1", "KaggleCRD", "RICORD"] #"BIMCV"
pneumonia_folders = ["Bacteria", "Fungi", "Virus", "Pneumonia", "Lung Opacity"]
pathogen_folders = ["Bacteria", "Fungi", "Virus", "Pneumonia", "Lung Opacity", "COVID-19", "Normal"]
origin_folder = "2_Raw_Seg"
dest_folder = "3_Images_Seg"

img_size = 400
cwd = os.getcwd()

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))


bot.send_message("-600800507", f"Iniciando separação de treino e teste")
# Remove destination dir, if present, just for start clean
if os.path.isdir(dest_folder):
  shutil.rmtree(dest_folder)

# Create intermediate folders, if they are not present
if not os.path.isdir(dest_folder):
  os.makedirs(dest_folder)

if not os.path.isdir(os.path.join(dest_folder, "masks")):
  os.makedirs(os.path.join(dest_folder, "masks"))

for target in ["train", "test"]:
  for pathogen in ["Opacity", "COVID-19", "Normal"]:
    pathogen_folder = os.path.join(dest_folder, target, pathogen)
    if not os.path.isdir(pathogen_folder):
      os.makedirs(pathogen_folder)

# Iterate over all source folders and pathogens to copy the relevant images
for folder in source_folders:
  for pathogen in pathogen_folders:
    pathogen_folder = os.path.join(cwd, origin_folder, folder, pathogen)
    pathogen_coded = "Opacity" if pathogen in pneumonia_folders else pathogen
    pathogen_folder = pathogen_folder +"/"
    
    print(pathogen_folder)
    print(os.path.isdir(pathogen_folder))
    
    if (os.path.isdir(pathogen_folder)):
      for (root, dirs, files) in os.walk(pathogen_folder, topdown = True):
        pid_list = {}
        for file in files:
          print(file)
          _, pid, offset = re.split("[P_]", os.path.splitext(file)[0])

          # If pid was not assigned to a group
          # Then random selected train/test following the desirable distribution
          if pid not in pid_list:
            prob = 0.8
            target_folder = "test" if random.uniform(0, 1) > prob else "train"
            pid_list[pid] = target_folder
          else:
            target_folder = pid_list[pid]
            
          print(file)
          # Copy image and rename file
          shutil.copy2(os.path.join(root, file), os.path.join(dest_folder, target_folder, pathogen_coded))
          new_filename = "%s_%s_%s_%s" % (folder, pathogen_coded, pid, offset)
          new_filename_ext = "%s%s" % (new_filename, os.path.splitext(file)[-1])
          os.rename(
            os.path.join(dest_folder, target_folder, pathogen_coded, file),
            os.path.join(dest_folder, target_folder, pathogen_coded, new_filename_ext),
          )

          # Well, let's already apply CLAHE to improve the CXR contrast and brightness
          img = cv2.imread(os.path.join(dest_folder, target_folder, pathogen_coded, new_filename_ext), cv2.IMREAD_COLOR)
          os.remove(os.path.join(dest_folder, target_folder, pathogen_coded, new_filename_ext))
          #img = clahe.apply(img)

          # Let's also resize the images so that all of them are standardize
          # Skip the image if it is too small
          print(img.shape)
          w, h = img.shape[:-1]
          if w < 250:
            continue

          img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)

          new_filename_ext = "%s%s" % (new_filename, ".png")
          cv2.imwrite(os.path.join(dest_folder, target_folder, pathogen_coded, new_filename_ext), img)

          # Check if there any mask for the specific
          # If yes, copy and resize it
          mask_file = "%s.png" % os.path.splitext(file)[0]
          mask_path = os.path.join(origin_folder, folder, "Masks", mask_file)
          if (os.path.isfile(mask_path)):
            mask_img = cv2.imread(mask_path, 0)
            mask_img = cv2.resize(mask_img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
            new_mask_filename = "%s_%s_%s_%s_%s.png" % (target_folder, folder, pathogen_coded, pid, offset)
            cv2.imwrite(os.path.join(dest_folder, "masks", new_mask_filename), mask_img)


gan = True

if (gan == True):

  source = '/home/dell/Documentos/covid-dissert/augmentation/Results/Pneumonia_Imgs/'
  files = os.listdir(source)
  dest = "/home/dell/Documentos/covid-dissert/3_Images_Seg/train/Opacity/"
  for file_name in random.sample(files, 500):
    shutil.copy2(os.path.join(source, file_name), dest)
  
  source = '/home/dell/Documentos/covid-dissert/augmentation/Results/COVID-19_Imgs/'
  files = os.listdir(source)
  dest = "/home/dell/Documentos/covid-dissert/3_Images_Seg/train/COVID-19/"
  for file_name in random.sample(files, 500):
    shutil.copy2(os.path.join(source, file_name), dest)


  source = '/home/dell/Documentos/covid-dissert/augmentation/Results/Normal_Imgs/'
  files = os.listdir(source)
  dest = "/home/dell/Documentos/covid-dissert/3_Images_Seg/train/Normal/"
  for file_name in random.sample(files, 500):
    shutil.copy2(os.path.join(source, file_name), dest)
