import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

def process_files(input_folder, output_folder, task_number):
    """
    Processes and resizes .nii.gz files in the input_folder and saves them to the output_folder.
    """
    for fold, task_folder in enumerate(os.listdir(input_folder)):
        task_path = os.path.join(input_folder, task_folder)
        if not os.path.isdir(task_path):
            continue
        
        # Determine new task folder name
        new_task_number = task_number + fold
        new_task_folder = f"Task{new_task_number}_CSMAbdominalSegAdjTo0Fold{fold + 1}"
        new_task_path = os.path.join(output_folder, new_task_folder)
        
        for root, dirs, files in os.walk(task_path):
            for file in tqdm(files, desc=f"Processing {task_folder}/{root.split('/')[-1]}"):
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, task_path)
                    new_relative_path = relative_path.replace(task_folder.split('_')[1], new_task_folder.split('_')[1])
                    output_path = os.path.join(new_task_path, new_relative_path)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Load the .nii.gz file
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    
                    if not "labels" in root:
                        # 進行data pre-processing
                        values, counts = np.unique(data, return_counts=True)
                        most_frequent_pixel = values[np.argmax(counts)]
                        #print(f"{file_path.split('_')[-2]} 最多的像素值是: {most_frequent_pixel}")
                        if most_frequent_pixel == -2000.0:
                            processed_data = data + 2000
                        elif most_frequent_pixel == -2048.0:
                            processed_data = data + 2048
                        else:
                            print("error")
                            exit(0)
                    else:
                        processed_data = data

                    # Save the resized image
                    resized_img = nib.Nifti1Image(processed_data, img.affine, img.header)
                    nib.save(resized_img, output_path)
                    #print(f"Resized and saved: {output_path}")

# Parameters
input_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out'
output_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out_AdjTo0'
task_number = 596

process_files(input_folder, output_folder, task_number)
