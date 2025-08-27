import os
import shutil
import pandas as pd
from tqdm import tqdm

def create_directory_structure(base_path):
    for i in range(1, 6):
        fold_folder = f"fold{i}"
        folders = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs', 'imagesVal', 'labelsVal']
        for folder in folders:
            os.makedirs(os.path.join(base_path, fold_folder, folder), exist_ok=True)

def load_number_mapping(excel_path):
    df = pd.read_excel(excel_path)
    # 確保 Old Number 是四位數的字符串格式
    df['New Number'] = df['New Number'].astype(str).str.zfill(4)
    return dict(zip(df['New Number'].astype(str), df['Old Number']))

def copy_files(source_path, destination_base, number_mapping, dcm_source, png_source):
    for task_folder in os.listdir(source_path):
        if task_folder.startswith("Task58"):
            fold_number = task_folder[-1]
            dest_fold = f"fold{fold_number}"
            
            for subset in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs', 'imagesVal', 'labelsVal']:
                source_subset = os.path.join(source_path, task_folder, subset)
                dest_subset = os.path.join(destination_base, dest_fold, subset)
                
                if not os.path.exists(source_subset):
                    continue  # 跳過不存在的文件夾
                
                for filename in tqdm(os.listdir(source_subset), desc=f"Processing {task_folder}/{subset}"):
                    if filename.endswith('.nii.gz'):
                        # 提取文件編號
                        if 'images' in subset:
                            file_number = filename.split('_')[-2]
                        else:  # labels
                            file_number = filename.split('_')[-1].split('.')[0]
                        
                        if file_number in number_mapping:
                            old_number = number_mapping[file_number]
                            
                            if 'images' in subset:
                                new_filename = f"{old_number}.dcm"
                                source_file = os.path.join(dcm_source, new_filename)
                            else:  # labels
                                new_filename = f"{old_number}.png"
                                source_file = os.path.join(png_source, new_filename)
                            
                            if os.path.exists(source_file):
                                shutil.copy(source_file, os.path.join(dest_subset, new_filename))
                            else:
                                print(f"Warning: {new_filename} not found in source folder.")
                        else:
                            print(f"Warning: Mapping not found for file number {file_number}")

def main():
    source_path = "raw_data"  # 原始數據路徑
    destination_base = "new_raw_data"  # 新的數據路徑
    excel_path = r"./renumbered_files.xlsx"
    dcm_source = r"./DCM_PNG/images"
    png_source = r"./DCM_PNG/labels"

    # 載入編號對照表
    number_mapping = load_number_mapping(excel_path)

    # 創建新的目錄結構
    create_directory_structure(destination_base)

    # 複製並重命名文件
    copy_files(source_path, destination_base, number_mapping, dcm_source, png_source)

if __name__ == "__main__":
    main()