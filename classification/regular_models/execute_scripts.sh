#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow

for i in 11 12 13 14 15 16 17 18 19 20 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  echo "counter: $i"
  
  cd /home/dell/Documentos/covid-dissert/
  
  python preprocess_Cohen-Kaggle-BIMCV-OCT.py && python preprocess_RICORD-RSNA.py &&
  cd /home/dell/Documentos/covid-dissert/classification/regular_models && python pt0_load_data.py -m VGG16 -i $i && 
  python pt1_transfer_learning_vgg16.py -m VGG16 -i $i && 
  python pt2_vgg16_finetune.py -m VGG16 -i $i && 
  python pt3_model_eval.py -i $i -m VGG16 && 
  python pt3_0_model_eval.py -i $i -m VGG16 && 
  python pt4_roc_curve.py -i $i -m VGG16
done
