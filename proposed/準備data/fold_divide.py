import os
import shutil
from sklearn.model_selection import KFold

# 設定資料夾路徑
base_dir = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/mendeley_dataset'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')
output_base_dir = os.path.join(base_dir, 'out')

# 創建輸出資料夾
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# 取得所有檔案名稱
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

# 確保檔案數量一致
assert len(images) == len(labels), "Images and Labels count do not match!"

# 獲取編號
ids = list(range(1, len(images) + 1))

# 進行5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(ids), 1):
    fold_dir = os.path.join(output_base_dir, f'Task{575 + fold}_AbdominalSegFold{fold}')
    images_tr_dir = os.path.join(fold_dir, 'imagesTr')
    images_ts_dir = os.path.join(fold_dir, 'imagesTs')
    labels_tr_dir = os.path.join(fold_dir, 'labelsTr')
    labels_ts_dir = os.path.join(fold_dir, 'labelsTs')
    
    # 創建子資料夾
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(images_ts_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    os.makedirs(labels_ts_dir, exist_ok=True)
    
    for idx in train_index:
        img_id = f"{ids[idx]:03d}"
        shutil.copy(os.path.join(image_dir, f"AbdominalSeg_{img_id}_0000.nii.gz"), 
                    os.path.join(images_tr_dir, f"AbdominalSegFold{fold}_{img_id}_0000.nii.gz"))
        shutil.copy(os.path.join(label_dir, f"AbdominalSeg_{img_id}_0000.nii.gz"), 
                    os.path.join(labels_tr_dir, f"AbdominalSegFold{fold}_{img_id}.nii.gz"))
    
    for idx in test_index:
        img_id = f"{ids[idx]:03d}"
        shutil.copy(os.path.join(image_dir, f"AbdominalSeg_{img_id}_0000.nii.gz"), 
                    os.path.join(images_ts_dir, f"AbdominalSegFold{fold}_{img_id}_0000.nii.gz"))
        shutil.copy(os.path.join(label_dir, f"AbdominalSeg_{img_id}_0000.nii.gz"), 
                    os.path.join(labels_ts_dir, f"AbdominalSegFold{fold}_{img_id}.nii.gz"))

print("5-fold cross-validation data preparation is complete.")
