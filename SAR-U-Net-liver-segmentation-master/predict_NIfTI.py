import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk
import time

import utils.metrics as m
# from model.unet.unet_model import UNet
# # from unet import UNet
# from model.FCN.FCN import *
# from model.attention_unet.attention_unet import AttU_Net
# from model.resunet.resunet import DeepResUNet
# from model.se_resunet_plus.se_resunet_plus import SeResUNet
# from model.se_p_attresunet.se_p_attresunet import SE_P_AttU_Net
from models.se_p_resunet.se_p_resunet import Se_PPP_ResUNet

from utils.preprocessing_utils import sitk2slices, sitk2labels
from utils.surface import Surface
# from utils.test_utils import (draw_contours, draw_many_slices, imwrite,
#                               remove_fragment)
from tqdm import tqdm
from utils.dataset import make_dataloader
import os
from surface_distance import metrics
from PIL import Image
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou(y_true, y_pred):
    intersection = np.sum((y_true & y_pred))
    union = np.sum((y_true | y_pred))
    return intersection / union

def hausdorff_distance95(y_true, y_pred):
    surface_distances = metrics.compute_surface_distances(y_true, y_pred, spacing_mm=(1, 1, 1))
    distances = metrics.compute_robust_hausdorff(surface_distances, 95)
    return distances

def visualize_segmentation(y_true, y_pred, class_name, output_path):
    # Create a color map for visualization
    visualization = np.zeros((*y_true.shape, 3), dtype=np.uint8)
    
    # TP: Green, TN: Black, FP: Red, FN: Yellow
    tp = np.logical_and(y_true, y_pred)
    tn = np.logical_and(~y_true, ~y_pred)
    fp = np.logical_and(~y_true, y_pred)
    fn = np.logical_and(y_true, ~y_pred)
    
    visualization[tp] = [0, 255, 0]       # Green
    visualization[tn] = [0, 0, 0]         # Black
    visualization[fp] = [255, 0, 0]       # Red
    visualization[fn] = [255, 255, 0]     # Yellow
    
    # Save the image as a PNG file
    image = Image.fromarray(visualization[:, :, 0])
    image.save(output_path)

def round_and_format(value, decimal_places):
    """
    四捨五入到指定的小數位數，並補0以確保有指定的小數位數。
    """
    return f"{value:.{decimal_places}f}"

if __name__ == '__main__':
    
    classes = {1: "Abdominal muscles", 2: "VAT"}
    
    # 讀取renumbered_files.xlsx文件
    try:
        df = pd.read_excel('renumbered_files.xlsx')
        print(f"成功讀取renumbered_files.xlsx，共{len(df)}行數據")
    except Exception as e:
        print(f"讀取renumbered_files.xlsx失敗: {e}")
        df = None

    try:
        for fold in range(1, 6):
            data_path = f'preprocessed_data/fold{fold}'
            model_path = f'final checkpoints/Se_PPP_ResUNet_ce_NoS/fold{fold}/best_model.pth'
            prediction_path = f'results/NIfTI'
            os.makedirs(prediction_path, exist_ok=True)
            os.makedirs(prediction_path + f'/fold{fold}', exist_ok=True)
            save_dir = os.path.join(prediction_path, f'fold{fold}')
    
            _infer_time_total = 0.0    # ← 新增：累計秒數
            _infer_case_total = 0      # ← 新增：累計病例數
            _infer_time_file = os.path.join(save_dir, "inference_time.txt")

            device = torch.device('cuda:0')
                # model =  UNet(1, 2).to(device)
                # model = Se_PPP_ResUNet(1,2,deep_supervision=False).to(device)
            # vgg_model = VGGNet()
            # model = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
            # model = AttU_Net(1,2).to(device)
            model = Se_PPP_ResUNet(1,3).to(device)
            # model = SeResUNet(1, 2, deep_supervision=False).to(device)
            # model = SE_P_AttU_Net(1,2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            sm = nn.Softmax(dim=1)

            dataloader = make_dataloader(data_path + "/imagesTs", data_path + "/labelsTs", batch=4, n_workers=4)
            
            with torch.no_grad():  # 添加這個來提高推理性能
                with tqdm(total=len(dataloader), desc='Predicting', unit='batch') as pbar:
                    for raw, labels, name in dataloader:
                        # ---------- timer start ----------
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t0 = time.perf_counter()

                        output = model(raw.to(device))
                        # print(output.shape) torch.Size([1, 2, 256, 256])
                        predictions = sm(output)
                        # print(prediction.shape)    #torch.Size([1, 2, 256, 256])
                        predictions = torch.argmax(predictions, dim=1) #返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                        # print(prediction.shape)     #torch.Size([1, 256, 256])
                        # print(prediction)
                        predictions = predictions.cpu().detach().numpy().astype(np.int32)
                        labels = labels.cpu().detach().numpy().astype(np.int32)


                        # ---------- timer end ------------
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _infer_time_total += time.perf_counter() - _t0
                        _infer_case_total += predictions.shape[0]

                        # 清理GPU記憶體（可選，如果仍有性能問題）
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        for i in range(predictions.shape[0]):
                            prediction = predictions[i][..., np.newaxis]
                            label = labels[i][..., np.newaxis]

                            #儲存成NIfTI
                            number = name["name"][i]
                            
                            # 在df的Old Number欄位找到對應number的那一列，並將New Number的值assign到number
                            if df is not None:
                                try:
                                    # 在Old Number欄位中查找對應的number
                                    matching_row = df[df['Old Number'] == int(number)]
                                    if not matching_row.empty:
                                        # 將New Number的值assign到number
                                        number = matching_row.iloc[0]['New Number']
                                        #print(f"重新編號: {name['name'][i]} -> {number}")
                                    else:
                                        print(f"警告: 在Excel中找不到Old Number為{number}的記錄")
                                except Exception as e:
                                    print(f"重新編號時發生錯誤: {e}")
                            
                            nii_path = os.path.join(save_dir, f'CSMAbdominalSegFold{fold}_{number:04}.nii.gz')
                            nii_image = nib.Nifti1Image(prediction, np.eye(4))
                            nib.save(nii_image, nii_path)

                        pbar.update()
            if _infer_case_total:
                sec_per_case = _infer_time_total / _infer_case_total
                print(f"Pure inference time: {_infer_time_total:.1f} s "
                        f"Total case amount: {_infer_case_total}, "
                    f"({_infer_time_total/_infer_case_total:.4f} s / case)")
                # --------  寫入 txt  --------
                with open(_infer_time_file, "w") as f:
                    f.write(f"total_seconds\t{_infer_time_total:.6f}\n")
                    f.write(f"cases\t{_infer_case_total}\n")
                    f.write(f"seconds_per_case\t{sec_per_case:.6f}\n")
    except Exception as e:
        print(e)
        pass