#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow

for i in 1 2 3 4 5
do
  echo "counter: $i"
  
  cd /home/dell/Documentos/covid-dissert/
  
  python preprocess_Cohen-Kaggle-RICORD-RSNA.py && python preprocess_OCT-BIMCV.py
  
  cd /home/dell/Documentos/covid-dissert/classification/regular_models && python pt0_load_data.py 
  
  python pt1_transfer_learning_vgg16.py && python pt2_vgg16_finetune.py && python pt3_model_eval.py -i $i && python pt4_roc_curve.py -i $i
done
