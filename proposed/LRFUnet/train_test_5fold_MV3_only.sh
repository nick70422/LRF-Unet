LRF_type='MV3_only'
LRF_r="128"
LRF_start_pos="0"

CSM=580
CSM_task_name='CSMAbdominalSeg'
mendeley=575
mendeley_task_name='AbdominalSeg'
CSMHuOtsu=605
CSMHuOtsu_task_name='CSMAbdominalSegHuOtsu'
dataset=$CSM
dataset_task_name=$CSM_task_name

#training
python3 run_training.py $(($dataset+1)) 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
python3 run_training.py $(($dataset+2)) 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
python3 run_training.py $(($dataset+3)) 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
python3 run_training.py $(($dataset+4)) 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
python3 run_training.py $(($dataset+5)) 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#python3 run_training.py 581 0 --pretrain supervised --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#python3 run_training.py 582 0 --pretrain supervised --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#python3 run_training.py 583 0 --pretrain supervised --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#python3 run_training.py 584 0 --pretrain supervised --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#python3 run_training.py 585 0 --pretrain supervised --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos --my_loss Dice+CE+SmoothL1
#testing
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+1))_${dataset_task_name}Fold1/imagesTs -o ./results/fold1 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+1)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+2))_${dataset_task_name}Fold2/imagesTs -o ./results/fold2 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+2)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+3))_${dataset_task_name}Fold3/imagesTs -o ./results/fold3 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+3)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+4))_${dataset_task_name}Fold4/imagesTs -o ./results/fold4 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+4)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+5))_${dataset_task_name}Fold5/imagesTs -o ./results/fold5 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+5)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 run_testing.py
echo Testing finished!!

#python3 ../dc_call.py Training and testing of LRF_AB was finished!