#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow

for i in  3 4 6  #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  echo "counter: $i"
  
  cd /home/dell/Documentos/covid-dissert/
  
  python preprocess_Cohen-Kaggle-RICORD-RSNA.py && python preprocess_OCT-BIMCV.py &&
  cd /home/dell/Documentos/covid-dissert/classification/regular_models && python pt0_load_data.py -m DenseNet -i $i && 
  python pt1_transfer_learning_vgg16.py -m DenseNet -i $i && 
  python pt2_vgg16_finetune.py -m DenseNet -i $i && 
  python pt3_model_eval.py -i $i -m DenseNet && 
  python pt4_roc_curve.py -i $i -m DenseNet
done
