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


from collections import OrderedDict
from typing import Tuple
import sys
import numpy as np
import torch
import time
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch,  to_cuda
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from generic_UNetPlusPlus import Generic_UNetPlusPlus
from generic_UNetPlusPlus_LoRA import Generic_UNetPlusPlus_LoRA
from generic_MobileV3UNetPlusPlus_LRF import Generic_MobileV3UNetPlusPlus_LRF
from generic_UNetPlusPlus_LRF import Generic_UNetPlusPlus_LRF
from generic_MobileV4UNetPlusPlus_LRF import Generic_MobileV4UNetPlusPlus_LRF
from generic_UNetPlusPlus_mobile_DW import Generic_MobileUNetPlusPlus_DW
from generic_UNetPlusPlus_Tucker import Generic_UNetPlusPlus_Tucker
from ConvTucker import ConvTucker
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import _LRScheduler
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from torchmetrics.image import StructuralSimilarityIndexMeasure
from loss import My_Loss, My_MultipleOutputLoss2

class nnUNetPlusPlusTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False,
                 #my own parameter
                 is_test=False, pretrain=None, finetune=None, LoRA_r=4, my_loss=None,
                 factorization=None, low_rank_conv=None, LRF_r=16, Tucker_threshold=0.95, start_pos=0, use_SE=False, use_CBAM=False, use_ASPP=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        #self.max_num_epochs = 2
        self.max_num_epochs = 50
        #self.max_num_epochs = 100
        #self.max_num_epochs = 500
        #self.max_num_epochs = 1000
        #self.max_num_epochs = 3000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_progress_bar = True
        # ---- training timer ----
        self._train_time_total = 0.0   # 累積秒數
        self._train_case_total = 0     # 累積影像（case）數  

        self.pin_memory = True

        self.is_test = is_test
        self.pretrain = pretrain
        self.finetune = finetune
        self.LoRA_r = LoRA_r
        self.my_loss = my_loss
        self.factorization = factorization
        self.low_rank_conv = low_rank_conv
        self.LRF_r = LRF_r
        self.Tucker_threshold = Tucker_threshold
        self.start_pos = start_pos
        self.use_SE = use_SE
        self.use_CBAM = use_CBAM
        self.use_ASPP = use_ASPP


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.batch_size = 1

            self.setup_DA_params()
            #if not self.my_loss in ['SSIM', 'SmoothL1+SSIM']:
            #    self.setup_DA_params()
            #else:
            #    self.my_setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            #self.ds_loss_weights = None
            # now wrap the loss
            if self.my_loss:
                self.loss = My_Loss(self.num_classes, self.my_loss, self.batch_dice)
            self.loss = My_MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        #LoRA debug
        #print(f"finetune: {self.finetune}")
        #print("self.factorization:", self.factorization)
        if self.finetune is None or self.finetune == 'full':
            self.network = Generic_UNetPlusPlus(self.num_input_channels, self.base_num_features, self.num_classes,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        elif self.finetune == 'LoRA':
            print('finetune: fintuning with LoRA')
            self.network = Generic_UNetPlusPlus_LoRA(self.num_input_channels, self.base_num_features, self.num_classes,
                                        len(self.net_num_pool_op_kernel_sizes),
                                        self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                        LoRA_location='encoder', LoRA_r=self.LoRA_r)
        if self.factorization == 'LRF':
            if self.low_rank_conv == 'AB':
                #print(f'start_pos: {self.start_pos}')
                self.network = Generic_MobileV3UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=self.LRF_r, LRF_start_pos=self.start_pos, use_SE=self.use_SE)
                #self.network = Generic_MobileV4UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self.num_classes,
                #                            len(self.net_num_pool_op_kernel_sizes),
                #                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                #                            dropout_op_kwargs,
                #                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                #                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                #                            LRF_r=self.LRF_r, LRF_start_pos=self.start_pos, use_SE=self.use_SE)
            if self.low_rank_conv == 'MV3_only':
                self.network = Generic_MobileV3UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            LRF_type='normal', LRF_r=self.LRF_r, LRF_start_pos=self.start_pos, use_SE=self.use_SE)
                #self.network = Generic_MobileV4UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self
            elif self.low_rank_conv == 'AB_dec':
                self.network = Generic_UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            LRF_r=self.LRF_r)
            elif '2Conv' in self.low_rank_conv:
                #print(f'start_pos: {self.start_pos}')
                self.network = Generic_MobileV3UNetPlusPlus_LRF(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            LRF_type=self.low_rank_conv, LRF_r=self.LRF_r, LRF_start_pos=self.start_pos, use_SE=self.use_SE, use_CBAM=self.use_CBAM, use_ASPP=self.use_ASPP)
            elif self.low_rank_conv == 'DW':
                print("DW used!!")
                self.network = Generic_MobileUNetPlusPlus_DW(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            LRF_type=self.low_rank_conv, LRF_r=self.LRF_r, LRF_start_pos=self.start_pos, use_SE=self.use_SE, use_CBAM=self.use_CBAM, use_ASPP=self.use_ASPP)
        elif self.factorization == 'Tucker':
            self.network = Generic_UNetPlusPlus_Tucker(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                            Tucker_start_pos=self.start_pos)
        if not self.pretrain is None and (not self.finetune is None or self.low_rank_conv == 'AB') and not self.is_test:
            self.load_pretrained()
        if self.factorization == 'Tucker':
            print("Processing Tucker factorization...")
            for module in self.network.modules():
                if isinstance(module, ConvTucker):
                    module.tucker_factorization(self.Tucker_threshold)

        print('\n\n\n\n Model parameters \n\n\n')
        model_parameters = filter(lambda p: p.requires_grad, (self.network).parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'No of params: {params}, {round(params / 1e6, 2)}M')
        print('\n\n\n\n')

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        #self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                 momentum=0.99, nesterov=True)
        #print("self.loss.parameters: ", list(self.loss.parameters()))
        #print("self.loss.Dice_CE_weights: ", list(self.loss.Dice_CE_weights))
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.loss.parameters()),  # 包含网络和损失函数的参数
            self.initial_lr, weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True
        )
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                               overwrite, validation_folder_name, debug, all_in_gpu, segmentation_export_kwargs)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = True,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes,
                                                                       use_sliding_window, step_size, use_gaussian,
                                                                       pad_border_mode, pad_kwargs, all_in_gpu, verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        measure = do_backprop          # 只在 training loop 計時

        if measure and torch.cuda.is_available():
            torch.cuda.synchronize()
        if measure:
            _t0 = time.perf_counter()
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        #show debug
        #print(f'data.shape: {data[0].shape}')
        #print(f'target.shape: {target[0].shape}')
        #initial_params = {}
        #from ConvLRF import ConvLRF
        #for name, module in self.network.named_modules():
        #    if isinstance(module, ConvLRF):
        #        initial_params[name] = {
        #            'A': module.A.clone().detach(),
        #            'B': module.B.clone().detach()
        #        }

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        b = data.shape[0]

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            #print('use fp16')
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                #print("Dice_CE_weights.grad (before clipping):", self.loss.Dice_CE_weights.grad)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
                #print("Dice_CE_weights (after update):", self.loss.Dice_CE_weights)
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        ## 檢查 A 和 B 是否更新
        #for name, module in self.network.named_modules():
        #    if isinstance(module, ConvLRF):
        #        updated_A = module.A
        #        updated_B = module.B
        #        if not torch.equal(initial_params[name]['A'], updated_A) or not torch.equal(initial_params[name]['B'], updated_B):
        #            print(f"Parameters updated in {name}")
        #        else:
        #            print(f"No parameter update in {name}")
#
        ## 更新 initial_params 為下一次迭代做準備
        #for name, module in self.network.named_modules():
        #    if isinstance(module, ConvLRF):
        #        initial_params[name]['A'] = module.A.clone().detach()
        #        initial_params[name]['B'] = module.B.clone().detach()

        if run_online_evaluation:
            #debug information
            #print("output.shape:", output[0].shape)
            #print("target.shape:", target[0].shape)
            self.run_online_evaluation(output, target)

        del target

        if measure:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._train_time_total += time.perf_counter() - _t0
            self._train_case_total +=  b  # 累積 case 數

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
            else:
                self.print_to_log_file("INFO: Requested fold %d but split file only has %d folds. I am now creating a "
                                       "random 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs
        print('memory ', torch.cuda.max_memory_allocated())
        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds

        #輸出計時結果
        os.path.join(self.output_folder, "")
        # ---- summarize timer ----
        if self._train_case_total:
            sec_per_case = self._train_time_total / self._train_case_total
            self.print_to_log_file(f"Total pure-training time: {self._train_time_total:.1f} s, "
                f"Total case amount: {self._train_case_total}, "
                f"({sec_per_case:.4f} s / case)")
            
        return ret


    def load_pretrained(self):
        pretrained_base = 'pretrained'
        #pretrained_base = 'UNetPP/pretrained'
        if self.pretrain == 'SSL':
            model_name = "SSL_simmim.pth"
        elif self.pretrain == 'supervised':
            model_name = "supervised_mendeley.model"
        pretrained_path = f'{pretrained_base}/{model_name}'
        print(f"Loading pretrained model from {pretrained_path}...")
        #device = torch.device('cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if self.pretrain == 'SSL':
            pretrained_dict = pretrained_dict['model']
            load_dict = {}
            for k, v in pretrained_dict.items():
                new_key = k.replace("model.", "")
                load_dict[new_key] = v
            pretrained_dict = load_dict
        elif self.pretrain == 'supervised':
            pretrained_dict = pretrained_dict['state_dict']

        if self.finetune == 'full':
            msg = self.network.load_state_dict(pretrained_dict, strict=True)
            print(msg)
        elif self.finetune == 'LoRA':
            #freeze
            freeze_li = []
            #debug_li = []
            for name, param in self.network.named_parameters():
                if 'conv_blocks_context' in name and not 'lora' in name and not str(self.network.num_pool) in name:
                    freeze_li.append(name)
                    param.requires_grad = False
                #if 'conv_blocks_context' in name:
                #    debug_li.append(name)

            msg = self.network.load_state_dict(pretrained_dict, strict=False)
            #print(msg)
            print('freeze_li:')
            for name in freeze_li:
                print(name)
            #print('debug_li:')
            #for name in debug_li:
            #    print(name)
            
    def plot_network_architecture(self):
        print("network plotting has been skipped in UNet++")

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def my_setup_DA_params(self):
        """
        设置不进行数据增强的参数。
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
        else:
            self.data_aug_params = default_2D_augmentation_params

        # 禁用所有数据增强操作
        self.data_aug_params['do_elastic'] = False
        self.data_aug_params['do_rotation'] = False
        self.data_aug_params['do_scaling'] = False
        self.data_aug_params['do_mirror'] = False
        self.data_aug_params['mirror_axes'] = ()
        self.data_aug_params['do_additive_brightness'] = False
        self.data_aug_params['do_gamma'] = False

        # 确保其他参数保持不变
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.data_aug_params["num_cached_per_thread"] = 2

        # 计算基本生成器的 patch 大小
        self.basic_generator_patch_size = np.array([0, 0])
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

if __name__ == '__main__':
    workdir = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/'
    plans_file = workdir + f'nnUNet_preprocessed/Task581_CSMAbdominalSegFold1/nnUNetPlansv2.1_plans_2D.pkl'
    dataset_directory = workdir + f'nnUNet_preprocessed/Task581_CSMAbdominalSegFold1'
    trainer = nnUNetPlusPlusTrainerV2(plans_file=plans_file, fold=0, output_folder="debug", dataset_directory=dataset_directory, batch_dice=True, stage=0, unpack_data=True, deterministic=False, fp16=True, factorization='LRF', low_rank_conv="AB", LRF_r=128)
    #trainer.initialize(False)
    trainer.initialize()