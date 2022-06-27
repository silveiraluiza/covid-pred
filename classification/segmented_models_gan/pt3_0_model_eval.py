#!/usr/bin/env python
# coding: utf-8

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

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    print("gpus existem")
    print(gpus)
  except RuntimeError as e:
    print(e)
    

parser = argparse.ArgumentParser(description='Processo avaliação de modelo')
parser.add_argument('-i','--index',  type=int , help='n da rodada do script', required=True)
parser.add_argument('-m','--model',  type=str , help='modelo', required=True)

args = parser.parse_args()

TELEBOT_TOKEN = "2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA"

bot = telebot.TeleBot("2058519653:AAG5Kf0Othtye8e13F5WPnBQQSdoCt47ifA")
bot.config['api_key'] = TELEBOT_TOKEN
bot.get_me()



def model_predict_(model, X_train):

  train1, train2, train3, train4, train5, train6 = np.array_split(X_train, 6)

  del(X_train)

  # Pred Train 
  Y_pred = model.predict(train1, batch_size=2)
  del(train1)

  Y_pred1 = model.predict(train2, batch_size=2)
  del(train2)

  Y_pred2 = model.predict(train3, batch_size=2)
  del(train3)

  Y_pred3 = model.predict(train4, batch_size=2)
  del(train4)

  Y_pred4 = model.predict(train5, batch_size=2)
  del(train5)

  Y_pred5 = model.predict(train6, batch_size=2)
  del(train6)

  y_pred = np.concatenate((Y_pred, Y_pred1, Y_pred2, Y_pred3, Y_pred4, Y_pred5))
  del(Y_pred, Y_pred1, Y_pred2, Y_pred3, Y_pred4, Y_pred5)

  return y_pred

def model_evaluation(y_pred, Y_true):
  y_pred = np.argmax(y_pred, axis = 1)
  y_true = np.argmax(Y_true, axis = 1)
  print('Confusion Matrix')
  df_cm = metrics.confusion_matrix(y_true, y_pred)
  print(df_cm)
  print('Classification Report')
  class_repo = metrics.classification_report(y_true, y_pred, output_dict=True, target_names = ["Opacity", "COVID-19", "Normal"])
  print(metrics.classification_report(y_true, y_pred, target_names = ["Opacity", "COVID-19", "Normal"]))
  print('F-Score')
  f_score = metrics.precision_recall_fscore_support(y_true, y_pred, average = "macro")
  print(f_score)
  print("Accuracy")
  acc = metrics.accuracy_score(y_true, y_pred)
  print(acc)

  return df_cm, class_repo, f_score, acc


def main():

  tf.keras.backend.clear_session()
  model_name = args.model
  ind = args.index


  ### Load Model

  
  if model_name.strip() == "VGG16":
    model_name = "cache/tl_vgg16_finetune.h5"
  
  elif model_name.strip() == "DenseNet":
    model_name = "cache/tl_densenet121_finetune.h5"

  elif model_name.strip() == "InceptionResNet":
    model_name = "cache/tl_inceptionresnet_finetune.h5"

  elif model_name.strip() == "ResNet152V2":
    model_name = "cache/tl_resnet152_finetune.h5"

  model = keras.models.load_model(model_name)



  ### Eval Test
  tf.keras.backend.clear_session()

  model = keras.models.load_model(model_name)
  
  ## Load Datasets
  X_test = np.load('input/X_test.npy')
  Y_test = np.load('input/Y_test.npy')

  start = time.time()
  Y_pred = model_predict_(model,X_test)
  end = time.time()
  del(X_test)

  print("Eval Test")
  df_cm_test, class_repo_test, f_score_test, acc_test = model_evaluation(Y_pred, Y_test)
  del(Y_test,model, Y_pred)
  final_test = end-start


  ### Eval Test 2 


  tf.keras.backend.clear_session()
  
  model = keras.models.load_model(model_name)
  
  # Load Data

  X_test2 = np.load('input/X_test2.npy')
  Y_test2 = np.load('input/Y_test2.npy')

  print(X_test2.shape)
  print(Y_test2.shape)
  
  
  Y_pred =  model_predict_(model,X_test2)
  del(X_test2)


  print("Eval Diff Test")
  df_cm_test2, class_repo_test2, f_score_test2, acc_test2 = model_evaluation(Y_pred, Y_test2)
  del(Y_test2, Y_pred)

  ### Load prior times

  with open('input/time_train.txt', 'r') as file:
      t1 = file.read().rstrip()
      
  with open('input/time_train_2.txt', 'r') as file:
      t2 = file.read().rstrip()


  tfinal = float(t1) + float(t2)
  
  ### Send Bot Msg

  time_hours = str(datetime.timedelta(seconds=tfinal))
  time_hours2 = str(datetime.timedelta(seconds=final_test))

  #bot.send_message("-600800507", f'Rede {model_name} - Treinamento Completo Finalizado em {time_hours}')
  bot.send_message("-600800507", f'Rede {model_name} - Predição Finalizada em {time_hours2}')
  bot.send_message("-600800507", f'Acurácia de teste: {acc_test}')


  ### Save Data


  

  old_csv = pd.read_csv("output/models_evaluation.csv", sep=';')

  
  csv = pd.read_csv("output/models_evaluation_temp.csv", sep=';')

  csv["tempo_pred"] = final_test
  csv["accuracy_test"] = [acc_test]
  csv["accuracy_test_diff"] = [acc_test2]

      
  csv["precision_COVID-19"] = [class_repo_test['COVID-19']["precision"]]
  csv["recall_COVID-19"] = [class_repo_test['COVID-19']["recall"]]
  csv["f1_COVID-19"] = [class_repo_test['COVID-19']["f1-score"]]
      
  csv["precision_Normal"] = [class_repo_test['Normal']["precision"]]
  csv["recall_Normal"] = [class_repo_test['Normal']["recall"]]
  csv["f1_Normal"] = [class_repo_test['Normal']["f1-score"]]
      
  csv["precision_Opacity"] = [class_repo_test['Opacity']["precision"]]
  csv["recall_Opacity"] = [class_repo_test['Opacity']["recall"]]
  csv["f1_Opacity"] = [class_repo_test['Opacity']["f1-score"]]

  csv["Class_Repo_Test"] = [class_repo_test]
  csv["Class_Repo_Test_diff"] = [class_repo_test2]

  csv["F_Score_Test"] = [f_score_test]
  csv["F_Score_Test_diff"] = [f_score_test2]
     
  csv["confusion_matrix_test"] = [df_cm_test]
  csv["confusion_matrix_test_diff"] = [df_cm_test2]


  new_csv = pd.concat([old_csv, csv])

  new_csv.to_csv("output/models_evaluation.csv", index=False, sep=';')


if __name__ == "__main__":
  main()