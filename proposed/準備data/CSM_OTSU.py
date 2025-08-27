import pandas as pd
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
from pydicom import dcmread
from skimage import filters, morphology, measure

def process_files(input_folder, output_folder, task_number):
    """
    Processes and resizes .nii.gz files in the input_folder and saves them to the output_folder.
    """

    excel_path = '準備data/CSM_dataset/renumbered_files.xlsx'
    df = pd.read_excel(excel_path)

    for fold, task_folder in enumerate(os.listdir(input_folder)):
        task_path = os.path.join(input_folder, task_folder)
        if not os.path.isdir(task_path):
            continue
        
        # Determine new task folder name
        new_task_number = task_number + fold
        new_task_folder = f"Task{new_task_number}_CSMAbdominalSegHuOtsuFold{fold + 1}"
        new_task_path = os.path.join(output_folder, new_task_folder)
        
        for root, dirs, files in os.walk(task_path):
            if len(files) == 0 or "labels" in root: continue
            for file in tqdm(files, desc=f"Processing {task_folder}/{root.split('/')[-1]}"):
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, task_path)
                    new_relative_path = relative_path.replace(task_folder.split('_')[1], new_task_folder.split('_')[1])
                    output_path = os.path.join(new_task_path, new_relative_path)
                    label_path = file_path.replace("_0000", "").replace("image", "label")
                    label_output_path = output_path.replace("_0000", "").replace("image", "label")
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                    
                    # Load the .nii.gz file and .dcm file
                    img = nib.load(file_path)
                    label = nib.load(label_path)
                    data = img.get_fdata()[:, :, 0]
                    label_data = label.get_fdata()[:, :, 0]
                    new_number = file.split("_")[1].split(".")[0]
                    old_number = df.set_index('New Number').at[int(new_number), 'Old Number']
                    dcm_file_path = f"準備data/CSM_dataset/images/{old_number}.dcm"
                    dcm = dcmread(dcm_file_path)

                    # 進行HU轉換及windowing
                    intercept = dcm.RescaleIntercept
                    slope = dcm.RescaleSlope
                    processed_data = data * slope + intercept
                    processed_data = np.clip(processed_data, -160, 240)

                    # Otsu找最大物件
                    th = filters.threshold_otsu(processed_data)
                    mask = processed_data >= th

                    # 剔除雜點、補洞
                    mask = morphology.remove_small_objects(mask, min_size=64)   # <64 px 直接丟
                    mask = morphology.remove_small_holes(mask, area_threshold=64)

                    # 連通元件標記
                    measure_label, num = measure.label(mask, connectivity=2, return_num=True)
                    props = measure.regionprops(measure_label)

                    # 擷取每個物件的「外接矩形」(min_row, min_col, max_row, max_col)
                    bboxes = [p.bbox for p in props]

                    # 找到面積最大的物件並裁切出來
                    largest = max(props, key=lambda p: p.area_bbox)
                    r0, c0, r1, c1 = largest.bbox
                    
                    processed_data = processed_data[r0:r1, c0:c1][..., np.newaxis]
                    processed_label_data = label_data[r0:r1, c0:c1][..., np.newaxis]

                    zoom_factors = [512 / processed_data.shape[0], 512 / processed_data.shape[1], 1]
                    processed_data = zoom(processed_data, zoom_factors, order=3)
                    zoom_factors_label = [512 / processed_label_data.shape[0], 512 / processed_label_data.shape[1], 1]
                    processed_label_data = zoom(processed_label_data, zoom_factors_label, order=0)  # order=0: nearest for label

                    # Save the processed image
                    processed_img = nib.Nifti1Image(processed_data, img.affine, img.header)
                    nib.save(processed_img, output_path)
                    processed_label = nib.Nifti1Image(processed_label_data, label.affine, label.header)
                    nib.save(processed_label, label_output_path)
                    #print(f"Resized and saved: {output_path}")

# Parameters
input_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out'
output_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out_HU_Otsu'
task_number = 606

process_files(input_folder, output_folder, task_number)
