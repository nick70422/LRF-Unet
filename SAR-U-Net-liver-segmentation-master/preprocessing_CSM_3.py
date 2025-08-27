import os
import shutil
import numpy as np
import pydicom
from skimage import exposure
from tqdm import tqdm

def create_directory_structure(base_path):
    for i in range(1, 6):
        fold_folder = f"fold{i}"
        folders = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs', 'imagesVal', 'labelsVal']
        for folder in folders:
            os.makedirs(os.path.join(base_path, fold_folder, folder), exist_ok=True)

def preprocess_image(dcm_path):
    # 讀取 DICOM 文件
    dcm = pydicom.dcmread(dcm_path)
    image = dcm.pixel_array.astype(np.int16)
    if dcm.RescaleSlope != 1:
        image = image * dcm.RescaleSlope
    if dcm.RescaleIntercept != 0:
        image = image + dcm.RescaleIntercept
    image = image.astype(np.float32)

    # 1. 將像素值限制在 [-240, 160] 之間
    image = np.clip(image, -240, 160)

    # 2. 應用直方圖均衡化
    image = exposure.equalize_hist(image)

    # 3. 將像素值標準化到 [0, 1] 之間
    image = (image - image.min()) / (image.max() - image.min())

    return image

def process_and_save_files(source_path, destination_base):
    for fold_folder in os.listdir(source_path):
        if fold_folder.startswith("fold"):
            for subset in ['imagesTr', 'imagesTs', 'imagesVal', 'labelsTr', 'labelsTs', 'labelsVal']:
                source_subset = os.path.join(source_path, fold_folder, subset)
                dest_subset = os.path.join(destination_base, fold_folder, subset)
                
                if not os.path.exists(source_subset):
                    continue  # 跳過不存在的文件夾
                
                for filename in tqdm(os.listdir(source_subset), desc=f"Processing {fold_folder}/{subset}"):
                    source_file = os.path.join(source_subset, filename)
                    
                    if filename.endswith('.dcm'):
                        # 處理 DCM 文件
                        processed_image = preprocess_image(source_file)
                        new_filename = filename.replace('.dcm', '.npy')
                        np.save(os.path.join(dest_subset, new_filename), processed_image)
                    elif filename.endswith('.png'):
                        # 直接複製 PNG 文件
                        shutil.copy(source_file, os.path.join(dest_subset, filename))
                    else:
                        print(f"Warning: Unexpected file type {filename}")

def main():
    source_path = "new_raw_data"  # 原始數據路徑
    destination_base = "preprocess_data"  # 新的數據路徑

    # 創建新的目錄結構
    create_directory_structure(destination_base)

    # 處理並保存文件
    process_and_save_files(source_path, destination_base)

if __name__ == "__main__":
    main()