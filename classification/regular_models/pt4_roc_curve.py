import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import telebot
import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import datetime
import sys
import argparse

parser = argparse.ArgumentParser(description='Processo avaliação de modelo')
parser.add_argument('-i','--index',  type=int , help='n da rodada do script', required=True)

args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    print("gpus existem")
    print(gpus)
  except RuntimeError as e:
    print(e)

def main():

  tf.keras.backend.clear_session()
  model_name = "cache/tl_vgg16_finetune_cd.h5"
  ind = int(args.index)

  model = keras.models.load_model(model_name)
  X_test = np.load('input/X_test.npy')
  Y_test = np.load('input/Y_test.npy')

  Y_pred = model.predict(X_test, batch_size = 16)

  i = 1
  fpr, tpr, thresholds = roc_curve(Y_test[:, i], Y_pred[:, i])
  data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
  df = pd.DataFrame(data)
  df.to_csv(f"output/nonsegmented_vgg16_roc_{ind}.csv", index = False)

  bot.send_message("-600800507", f'Rede {model_name} - Curva ROC salva')


if __name__ == "__main__":
  main()