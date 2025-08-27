import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk

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
    results = []

    try:
        for fold in range(1, 6):
            data_path = f'preprocess_data/fold{fold}'
            model_path = f'final checkpoints/Se_PPP_ResUNet_ce_NoS/fold{fold}/best_model.pth'
            prediction_path = f'results/'
            os.makedirs(prediction_path + f'fold{fold}', exist_ok=True)

            fold_results = {'Fold': fold}
            metrics_sum = {f'Dice_{class_name}': 0 for class_name in classes.values()}
            metrics_sum.update({f'IoU_{class_name}': 0 for class_name in classes.values()})
            metrics_sum.update({f'Hausdorff95_{class_name}': 0 for class_name in classes.values()})

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
                        output = model(raw.to(device))
                        # print(output.shape) torch.Size([1, 2, 256, 256])
                        predictions = sm(output)
                        # print(prediction.shape)    #torch.Size([1, 2, 256, 256])
                        predictions = torch.argmax(predictions, dim=1) #返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                        # print(prediction.shape)     #torch.Size([1, 256, 256])
                        # print(prediction)
                        predictions = predictions.cpu().detach().numpy().astype(np.int32)
                        labels = labels.cpu().detach().numpy().astype(np.int32)

                        for i in range(predictions.shape[0]):
                            case_dir = os.path.join(prediction_path, f'fold{fold}', name['name'][i])
                            os.makedirs(case_dir, exist_ok=True)

                            prediction = predictions[i][..., np.newaxis]
                            label = labels[i][..., np.newaxis]
                            for class_value, class_name in classes.items():
                                y_true = (label == class_value)
                                y_pred = (prediction == class_value)

                                dice = dice_coefficient(y_true, y_pred)
                                iou_value = iou(y_true, y_pred)
                                hd95 = hausdorff_distance95(y_true, y_pred)

                                metrics_sum[f'Dice_{class_name}'] += dice
                                metrics_sum[f'IoU_{class_name}'] += iou_value
                                metrics_sum[f'Hausdorff95_{class_name}'] += hd95

                                # 可視化結果
                                output_path = os.path.join(case_dir, f'{class_name}.png')
                                visualize_segmentation(y_true, y_pred, class_name, output_path)

                        pbar.update()

                    # 計算平均值
                    num_files = len(dataloader.dataset.filelist)
                    for key in metrics_sum.keys():
                        metrics_sum[key] = round_and_format(metrics_sum[key] / num_files, 4)

                    # 添加到 fold 結果中
                    fold_results.update(metrics_sum)

                    # 計算所有類別的平均指標（不包括背景）
                    dice_avg = np.mean([float(fold_results[f'Dice_{class_name}']) for class_name in classes.values()])
                    iou_avg = np.mean([float(fold_results[f'IoU_{class_name}']) for class_name in classes.values()])
                    hd95_avg = np.mean([float(fold_results[f'Hausdorff95_{class_name}']) for class_name in classes.values()])

                    fold_results['Dice_Average'] = round_and_format(dice_avg, 4)
                    fold_results['IoU_Average'] = round_and_format(iou_avg, 4)
                    fold_results['Hausdorff95_Average'] = round_and_format(hd95_avg, 4)

                    results.append(fold_results)
        
        # 計算所有fold的平均
        average_results = {'Fold': 'Average'}
        for metric in results[0].keys():
            if metric != 'Fold':
                metric_values = [float(fold_result[metric]) for fold_result in results if fold_result['Fold'] != 'Average']
                average_results[metric] = round_and_format(np.mean(metric_values), 4)
        
        results.append(average_results)
    except Exception as e:
        print(e)
        pass

# 創建DataFrame並保存到Excel
df = pd.DataFrame(results)
# 创建一个新的工作簿和工作表
wb = Workbook()
ws = wb.active

# 将DataFrame数据写入工作表
for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        
        # 设置字体和大小
        cell.font = Font(name='Calibri', size=18)
        
        # 设置对齐方式
        cell.alignment = Alignment(horizontal='center', vertical='center')

# 调整列宽
for column in ws.columns:
    max_length = 0
    column_letter = column[0].column_letter
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(cell.value)
        except:
            pass
    adjusted_width = (max_length + 2) * 1.2
    ws.column_dimensions[column_letter].width = adjusted_width

# 保存Excel文件
wb.save(f'{prediction_path}/testing_result.xlsx')

# 打印结果
print(df)