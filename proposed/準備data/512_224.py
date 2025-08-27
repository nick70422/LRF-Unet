import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resize_labels(source_folder, target_folder, start_idx, end_idx, target_shape=(224, 224, 1)):
    # 確保目標資料夾存在
    os.makedirs(target_folder, exist_ok=True)

    # 獲取所有符合條件的檔案
    nii_files = [f for f in os.listdir(source_folder) if f.endswith('.nii.gz') and nib.load(os.path.join(source_folder, f)).shape == (512, 512, 1)]

    # 檔案重新命名的格式，移除 _0000
    filename_format = 'AbdominalSeg224_{:03d}.nii.gz'

    # 開始處理檔案，編號從 start_idx 到 end_idx
    for idx, file_name in enumerate(nii_files, start=start_idx):
        if idx > end_idx:
            break

        # 讀取.nii.gz檔案
        img = nib.load(os.path.join(source_folder, file_name))
        data = img.get_fdata()

        # 檢查數據中的唯一值
        unique_values = np.unique(data)
        if not np.all(np.isin(unique_values, [0, 1, 2, 3, 4, 5, 6])):
            print(f"警告: 檔案 {file_name} 包含不在 0-6 範圍內的值: {unique_values}")
            continue

        # 強制數據類型為整數
        data = data.astype(np.int32)

        # 計算縮放比例
        scale_factors = np.array(target_shape) / np.array(data.shape)

        # Resize影像資料，對於標籤圖片，應使用最近鄰插值 (order=0)
        resized_data = zoom(data, scale_factors, order=0)  # 使用最近鄰插值 (order=0)

        # 修正插值後的數據，確保為整數
        resized_data = np.round(resized_data).astype(np.int32)

        # 檢查調整大小後的數據中的唯一值
        resized_unique_values = np.unique(resized_data)
        if not np.all(np.isin(resized_unique_values, [0, 1, 2, 3, 4, 5, 6])):
            print(f"警告: 調整大小後的檔案 {file_name} 包含不在 0-6 範圍內的值: {resized_unique_values}")
            continue

        # 建立新的NIfTI影像
        new_img = nib.Nifti1Image(resized_data, img.affine, img.header)

        # 生成新檔名
        new_file_name = filename_format.format(idx)

        # 儲存至目標資料夾
        nib.save(new_img, os.path.join(target_folder, new_file_name))

    print(f"所有檔案已從 {start_idx} 號處理到 {end_idx} 號。")

# 處理 labelsTr 資料夾
source_folder_tr = r'D:\Nick\medical\code\DL_models_benchmark-master\data\nnUNet_raw\nnUNet_raw_data\Task575_AbdominalSeg\labelsTr'
target_folder_tr = r'D:\Nick\medical\code\DL_models_benchmark-master\data\nnUNet_raw\nnUNet_raw_data\Task576_AbdominalSeg224\labelsTr'
resize_labels(source_folder_tr, target_folder_tr, start_idx=1, end_idx=435)

# 處理 labelsTs 資料夾
source_folder_ts = r'D:\Nick\medical\code\DL_models_benchmark-master\data\nnUNet_raw\nnUNet_raw_data\Task575_AbdominalSeg\labelsTs'
target_folder_ts = r'D:\Nick\medical\code\DL_models_benchmark-master\data\nnUNet_raw\nnUNet_raw_data\Task576_AbdominalSeg224\labelsTs'
resize_labels(source_folder_ts, target_folder_ts, start_idx=436, end_idx=575)
