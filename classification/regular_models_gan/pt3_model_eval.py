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

def model_predict_train(model, X_train):

  train1, train2, train3, train4, train5, train6 = np.array_split(X_train, 6)

  train1 = np.stack((train1,) * 3, axis= -1)
  train2 =  np.stack((train2,) * 3, axis= -1)
  train3 =  np.stack((train3,) * 3, axis= -1)
  train4 = np.stack((train4,) * 3, axis= -1)
  train5 = np.stack((train5,) * 3, axis= -1)
  train6 = np.stack((train6,) * 3, axis= -1)

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

  X_train = np.load('input/X_train.npy')
  Y_train = np.load('input/Y_train.npy')

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


  ### Eval Train
  Y_pred = model_predict_train(model, X_train)
  del(X_train)

  print("Eval Train")
  df_cm_train, class_repo_train, f_score_train, acc_train = model_evaluation(Y_pred, Y_train)
  del(Y_train,Y_pred, model)

  ### Eval Val
  tf.keras.backend.clear_session()

  model = keras.models.load_model(model_name)
  
  ## Load Datasets
  X_val = np.load('input/X_val.npy')
  Y_val = np.load('input/Y_val.npy')
  
  Y_pred = model_predict_(model,X_val)
  del(X_val)

  print("Eval Val")
  df_cm_val, class_repo_val, f_score_val, acc_val = model_evaluation(Y_pred, Y_val)
  del(Y_val, model, Y_pred)

  
  ### Load prior times

  with open('input/time_train.txt', 'r') as file:
      t1 = file.read().rstrip()
      
  with open('input/time_train_2.txt', 'r') as file:
      t2 = file.read().rstrip()


  tfinal = float(t1) + float(t2)
  
  ### Send Bot Msg

  time_hours = str(datetime.timedelta(seconds=tfinal))
  #time_hours2 = str(datetime.timedelta(seconds=final_test))

  bot.send_message("-600800507", f'Rede {model_name} - Treinamento Completo Finalizado em {time_hours}')
  #bot.send_message("-600800507", f'Rede {model_name} - Predição Finalizada em {time_hours2}')
  #bot.send_message("-600800507", f'Acurácia de teste: {acc_test}')


  ### Save Data


  n_epochs = 150
  drop = 0.5
  IMG_SIZE = (300,300,3)
  

  #old_csv = pd.read_csv("output/models_evaluation.csv", sep=';')

  
  csv = pd.DataFrame()
  csv["model"] = [model_name]
  csv["base"] = ["Cohen-RICORD-Kaggle-RSNA"]
  csv["tempo_treino"] = tfinal
  csv["tempo_pred"] = [None] #final_test
  csv["full_retrain"] = True
  csv["accuracy_test"] = [None] #[acc_test]
  csv["accuracy_test_diff"] = [None] #[acc_test2]
  csv["accuracy_train"] = [acc_train]
  csv["accuracy_val"] = [acc_val]
  csv["img_size"] = [IMG_SIZE]
  csv["seed"] = [587]
      
  csv["precision_COVID-19"] = [None] #[class_repo_test['COVID-19']["precision"]]
  csv["recall_COVID-19"] = [None] #[class_repo_test['COVID-19']["recall"]]
  csv["f1_COVID-19"] = [None] #[class_repo_test['COVID-19']["f1-score"]]
      
  csv["precision_Normal"] = [None] #[class_repo_test['Normal']["precision"]]
  csv["recall_Normal"] = [None] #[class_repo_test['Normal']["recall"]]
  csv["f1_Normal"] = [None] #[class_repo_test['Normal']["f1-score"]]
      
  csv["precision_Opacity"] = [None] #[class_repo_test['Opacity']["precision"]]
  csv["recall_Opacity"] = [None] #[class_repo_test['Opacity']["recall"]]
  csv["f1_Opacity"] = [None] #[class_repo_test['Opacity']["f1-score"]]

  csv["Class_Repo_Train"] = [class_repo_train]
  csv["Class_Repo_Val"] = [class_repo_val]
  csv["Class_Repo_Test"] = [None] #[class_repo_test]
  csv["Class_Repo_Test_diff"] = [None] #[class_repo_test2]

  csv["F_Score_Train"] = [f_score_train]
  csv["F_Score_Val"] = [f_score_val]
  csv["F_Score_Test"] = [None] #[f_score_test]
  csv["F_Score_Test_diff"] = [None] #[f_score_test2]
  
      
  csv["confusion_matrix_train"] = [df_cm_train]    
  csv["confusion_matrix_val"] = [df_cm_val]    
  csv["confusion_matrix_test"] = [None] #[df_cm_test]
  csv["confusion_matrix_test_diff"] = [None] #[df_cm_test2]

  csv["epochs"] = [n_epochs]
  csv["augmentation"] = ["simple + GAN"]
  csv["dropout"] = [drop]
  csv["opt"] = "Adam"
      
  csv["base_learn_rate"] = ['0.001']
  csv["id"] = ind

  #new_csv = pd.concat([old_csv, csv])

  csv.to_csv("output/models_evaluation_temp.csv", index=False, sep=';')


if __name__ == "__main__":
  main()