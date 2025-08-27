import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from surface_distance import metrics
from PIL import Image

# 設定資料夾路徑
base_dir = '/mnt/d/Nick/medical/code/DL_models_benchmark-master/UNetPP'

start_task_num = 586
task_name = "CSMAbdominalSeg224"

folds = {
    'fold1': {
        'prediction_dir': f'{base_dir}/results/fold1/',
        'label_dir': f'{base_dir}/../data/nnUNet_raw/nnUNet_raw_data/Task{start_task_num+0}_{task_name}Fold1/labelsTs/'
    },
    'fold2': {
        'prediction_dir': f'{base_dir}/results/fold2/',
        'label_dir': f'{base_dir}/../data/nnUNet_raw/nnUNet_raw_data/Task{start_task_num+1}_{task_name}Fold2/labelsTs/'
    },
    'fold3': {
        'prediction_dir': f'{base_dir}/results/fold3/',
        'label_dir': f'{base_dir}/../data/nnUNet_raw/nnUNet_raw_data/Task{start_task_num+2}_{task_name}Fold3/labelsTs/'
    },
    'fold4': {
        'prediction_dir': f'{base_dir}/results/fold4/',
        'label_dir': f'{base_dir}/../data/nnUNet_raw/nnUNet_raw_data/Task{start_task_num+3}_{task_name}Fold4/labelsTs/'
    },
    'fold5': {
        'prediction_dir': f'{base_dir}/results/fold5/',
        'label_dir': f'{base_dir}/../data/nnUNet_raw/nnUNet_raw_data/Task{start_task_num+4}_{task_name}Fold5/labelsTs/'
    }
}

# 類別字典
#classes = {1: "SAT", 2: "VAT", 3: "Organs", 4: "Spine", 5: "Spine muscles", 6: "Abdominal muscles"} #mendeley
classes = {1: "Abdominal muscles", 2: "VAT"} #CSM

# 可視化資料夾路徑
visualization_base_dir = f'{base_dir}/results/visualize/'

# 創建可視化資料夾及其子資料夾
os.makedirs(visualization_base_dir, exist_ok=True)
for fold_name in folds.keys():
    os.makedirs(os.path.join(visualization_base_dir, fold_name), exist_ok=True)

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

# 保存結果的字典
results = []

try:
    # 遍歷每個fold
    for fold_name, paths in folds.items():
        prediction_dir = paths['prediction_dir']
        label_dir = paths['label_dir']
        visualization_dir = os.path.join(visualization_base_dir, fold_name)

        # 獲取所有預測和標籤檔案的名稱
        prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.nii.gz')]
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]

        fold_results = {'Fold': fold_name}
        metrics_sum = {f'Dice_{class_name}': 0 for class_name in classes.values()}
        metrics_sum.update({f'IoU_{class_name}': 0 for class_name in classes.values()})
        metrics_sum.update({f'Hausdorff95_{class_name}': 0 for class_name in classes.values()})

        print(f"Processing {fold_name}...")

        # 計算每個檔案的指標
        for pred_file in tqdm(prediction_files):
            pred_path = os.path.join(prediction_dir, pred_file)
            label_path = os.path.join(label_dir, pred_file)

            if not os.path.exists(label_path):
                continue
            
            # 加載影像資料
            prediction_img = nib.load(pred_path)
            label_img = nib.load(label_path)

            prediction_data = prediction_img.get_fdata().astype(np.int32)
            label_data = label_img.get_fdata().astype(np.int32)

            # Create a directory for the current testing case
            case_id = pred_file.split('_')[-1].split('.')[0]
            case_dir = os.path.join(visualization_dir, f'{task_name}_{case_id}')
            os.makedirs(case_dir, exist_ok=True)

            # 計算每個類別的指標
            for class_value, class_name in classes.items():
                y_true = (label_data == class_value)
                y_pred = (prediction_data == class_value)

                dice = dice_coefficient(y_true, y_pred)
                iou_value = iou(y_true, y_pred)
                hd95 = hausdorff_distance95(y_true, y_pred)

                metrics_sum[f'Dice_{class_name}'] += dice
                metrics_sum[f'IoU_{class_name}'] += iou_value
                metrics_sum[f'Hausdorff95_{class_name}'] += hd95

                # 可視化結果
                output_path = os.path.join(case_dir, f'{class_name}.png')
                visualize_segmentation(y_true, y_pred, class_name, output_path)

        # 計算平均值
        num_files = len(prediction_files)
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
except:
    pass

# 創建DataFrame並保存到Excel
df = pd.DataFrame(results)
df.to_excel(f'{base_dir}/results/testing_result.xlsx', index=False)

# 打印結果
print(df)
