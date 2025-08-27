import os
import shutil
import numpy as np
from sklearn.model_selection import KFold
from collections import OrderedDict
import pickle

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_splits(dataset_keys, splits_file):
    splits = []
    all_keys_sorted = np.sort(dataset_keys)
    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    save_pickle(splits, splits_file)
    return splits

def move_to_validation(source_path, fold=0):
    # 獲取所有文件的鍵（編號）
    all_keys = set()
    for task_folder in os.listdir(source_path):
        if task_folder.startswith("Task58"):
            images_tr_path = os.path.join(source_path, task_folder, 'imagesTr')
            for filename in os.listdir(images_tr_path):
                if filename.endswith('.nii.gz'):
                    file_number = filename.split('_')[-2]
                    all_keys.add(file_number)

    # 創建或加載分割
    splits_file = os.path.join(source_path, "splits_final.pkl")
    if not os.path.isfile(splits_file):
        splits = create_splits(list(all_keys), splits_file)
    else:
        splits = load_pickle(splits_file)

    # 獲取驗證集的鍵
    val_keys = set(splits[fold]['val'])

    for task_folder in os.listdir(source_path):
        if task_folder.startswith("Task58"):
            # 創建驗證集文件夾
            os.makedirs(os.path.join(source_path, task_folder, 'imagesVal'), exist_ok=True)
            os.makedirs(os.path.join(source_path, task_folder, 'labelsVal'), exist_ok=True)

            # 處理 imagesTr
            images_tr_path = os.path.join(source_path, task_folder, 'imagesTr')
            images_val_path = os.path.join(source_path, task_folder, 'imagesVal')
            for filename in os.listdir(images_tr_path):
                if filename.endswith('.nii.gz'):
                    file_number = filename.split('_')[-2]
                    if file_number in val_keys:
                        shutil.move(os.path.join(images_tr_path, filename),
                                    os.path.join(images_val_path, filename))

            # 處理 labelsTr
            labels_tr_path = os.path.join(source_path, task_folder, 'labelsTr')
            labels_val_path = os.path.join(source_path, task_folder, 'labelsVal')
            for filename in os.listdir(labels_tr_path):
                if filename.endswith('.nii.gz'):
                    file_number = filename.split('_')[-1].split('.')[0]
                    if file_number in val_keys:
                        shutil.move(os.path.join(labels_tr_path, filename),
                                    os.path.join(labels_val_path, filename))

    print(f"Validation set (fold {fold}) has been created and files have been moved.")

if __name__ == "__main__":
    source_path = "raw_data"  # 請根據實際情況修改路徑
    move_to_validation(source_path, fold=0)