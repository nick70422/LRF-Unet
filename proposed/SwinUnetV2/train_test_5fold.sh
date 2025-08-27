python3 run_training.py 581 0
#python3 ../dc_call.py Fold1 finished!
python3 run_training.py 582 0
#python3 ../dc_call.py Fold2 finished!
python3 run_training.py 583 0
#python3 ../dc_call.py Fold3 finished!
python3 run_training.py 584 0
#python3 ../dc_call.py Fold4 finished!
python3 run_training.py 585 0
#python3 ../dc_call.py Fold5 finished!

python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task581_CSMAbdominalSegFold1/imagesTs -o ./results/fold1 -tr SwinUNetV2TrainerV2 -m 2d -p nnUNetPlansv2.1 -t 581 -chk model_best -f 0
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task582_CSMAbdominalSegFold2/imagesTs -o ./results/fold2 -tr SwinUNetV2TrainerV2 -m 2d -p nnUNetPlansv2.1 -t 582 -chk model_best -f 0
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task583_CSMAbdominalSegFold3/imagesTs -o ./results/fold3 -tr SwinUNetV2TrainerV2 -m 2d -p nnUNetPlansv2.1 -t 583 -chk model_best -f 0
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task584_CSMAbdominalSegFold4/imagesTs -o ./results/fold4 -tr SwinUNetV2TrainerV2 -m 2d -p nnUNetPlansv2.1 -t 584 -chk model_best -f 0
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task585_CSMAbdominalSegFold5/imagesTs -o ./results/fold5 -tr SwinUNetV2TrainerV2 -m 2d -p nnUNetPlansv2.1 -t 585 -chk model_best -f 0
python3 run_testing.py
#python3 ../dc_call.py Testing finished!!