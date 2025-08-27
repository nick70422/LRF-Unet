import os
import json
import shutil

# 定義路徑
json_file = r'C:\Nick\研究\medical\code\DL_models_benchmark-master\準備data\CSM_dataset\5_fold_cross_validation.json'
images_nifti_folder = r'C:\Nick\研究\medical\code\DL_models_benchmark-master\準備data\CSM_dataset\images_nifti'
labels_nifti_folder = r'C:\Nick\研究\medical\code\DL_models_benchmark-master\準備data\CSM_dataset\labels_nifti'
output_folder = r'D:\Nick\medical\code\DL_models_benchmark-master\準備data\CSM_dataset\out'

# 讀取JSON檔案
with open(json_file, 'r') as f:
    folds = json.load(f)

# 創建輸出資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 複製檔案到相應的資料夾
for fold_idx, fold in enumerate(folds, start=1):
    task_folder = os.path.join(output_folder, f'Task58{fold_idx}_CSMAbdominalSegFold{fold_idx}')
    imagesTr_folder = os.path.join(task_folder, 'imagesTr')
    imagesTs_folder = os.path.join(task_folder, 'imagesTs')
    labelsTr_folder = os.path.join(task_folder, 'labelsTr')
    labelsTs_folder = os.path.join(task_folder, 'labelsTs')

    # 創建資料夾
    for folder in [imagesTr_folder, imagesTs_folder, labelsTr_folder, labelsTs_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 複製training set影像和標籤
    for img in fold['train_images']:
        img_filename = f'CSMAbdominalSegFold{fold_idx}_{img:04d}_0000.nii.gz'
        label_filename = f'CSMAbdominalSegFold{fold_idx}_{img:04d}.nii.gz'
        shutil.copy(os.path.join(images_nifti_folder, f'{img:04d}.nii.gz'), os.path.join(imagesTr_folder, img_filename))
        shutil.copy(os.path.join(labels_nifti_folder, f'{img:04d}.nii.gz'), os.path.join(labelsTr_folder, label_filename))
    
    # 複製testing set影像和標籤
    for img in fold['test_images']:
        img_filename = f'CSMAbdominalSegFold{fold_idx}_{img:04d}_0000.nii.gz'
        label_filename = f'CSMAbdominalSegFold{fold_idx}_{img:04d}.nii.gz'
        shutil.copy(os.path.join(images_nifti_folder, f'{img:04d}.nii.gz'), os.path.join(imagesTs_folder, img_filename))
        shutil.copy(os.path.join(labels_nifti_folder, f'{img:04d}.nii.gz'), os.path.join(labelsTs_folder, label_filename))

print('Files have been successfully copied to the respective folders.')
