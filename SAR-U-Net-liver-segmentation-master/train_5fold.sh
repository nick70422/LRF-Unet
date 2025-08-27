#!/bin/bash

for i in {1..5}
do
    echo "Processing fold$i"
    python3 main.py --base_dir preprocess_data --fold $i
    #python3 dc_call.py Fold $i finished
done

echo "All folds processed"
#python3 dc_call.py The training of SAR-U-Net were finished!!