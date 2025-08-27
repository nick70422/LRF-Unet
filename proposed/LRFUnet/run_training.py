#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnUNetPlusPlusTrainerV2 import nnUNetPlusPlusTrainerV2

#from nnunet.training.network_training.nnUNetTrainerV3 import nnUNetTrainerV3


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("network")
    #parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--pretrain", default=None,
                        help="pretrain type, support SSL(simmim) and supervised")
    parser.add_argument("--finetune", default=None,
                        help="finetune type, if default training from scratch, support full and LoRA")
    parser.add_argument("--LoRA_r", type=int, default=4,
                        help="only used in LoRA finetune, setting the r value of LoRA")
    parser.add_argument("--my_loss", default=None,
                        help="Use when you was pretraining, default cross entropy + dice loss. L1, L2, SmoothL1, SSIM, SmoothL1+SSIM is support")
    parser.add_argument("--factorization", default=None,
                        help="The type of factorization used.")
    parser.add_argument("--low_rank_conv", default=None,
                        help="Use custom low rank factorization convolution")
    parser.add_argument("--LRF_r", default=16,
                        help="The r value of LRFConv")
    parser.add_argument("--Tucker_threshold", default=0.95,
                        help="The threshold of r in Tucker factorization")
    parser.add_argument("--start_pos", default=0,
                        help="the layer number that start adding LRFConv, start from 0")
    parser.add_argument("--use_SE", default=False, action="store_true",
                        help="use SE block in StackedConv blocks or not")
    parser.add_argument("--use_CBAM", default=False, action="store_true",
                        help="use CBAM block in StackedConv blocks or not")
    parser.add_argument("--use_ASPP", default=False, action="store_true",
                        help="use ASPP in Bottleneck or not")

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    validation_only = args.validation_only
    plans_identifier = args.p

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    #my own parameter
    finetune = args.finetune
    pretrain = args.pretrain
    LoRA_r = args.LoRA_r if finetune == 'LoRA' else None
    my_loss = args.my_loss
    factorization = args.factorization
    low_rank_conv = args.low_rank_conv
    LRF_r = int(args.LRF_r)
    Tucker_threshold = float(args.Tucker_threshold)
    start_pos = int(args.start_pos)
    use_SE = args.use_SE
    use_CBAM = args.use_CBAM
    use_ASPP = args.use_ASPP

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    workdir = '/mnt/c/Nick/研究/medical/code/交接程式/proposed/data/'
    plans_file = workdir + f'nnUNet_preprocessed/{task}/nnUNetPlansv2.1_plans_2D.pkl'
    output_folder_name = workdir + f'nnUNet_trained_models/nnUNet/2d/{task}/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1' 
    dataset_directory = workdir + f'nnUNet_preprocessed/{task}'
    batch_dice = True 
    stage = 0
    trainer_class = nnUNetPlusPlusTrainerV2 


    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision,
                            pretrain=pretrain, finetune=finetune, LoRA_r=LoRA_r, my_loss=my_loss,
                            factorization=factorization, low_rank_conv=low_rank_conv, LRF_r=LRF_r, Tucker_threshold=Tucker_threshold, start_pos=start_pos, use_SE=use_SE, use_CBAM=use_CBAM, use_ASPP=use_ASPP)


    trainer.initialize(not validation_only)



    if not validation_only:
        if args.continue_training:
            trainer.load_latest_checkpoint()
        trainer.run_training()
    else:
        trainer.load_latest_checkpoint(train=False)

    trainer.network.eval()

    # predict validation
    trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)



if __name__ == "__main__":
    main()
