import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resize_image(image, new_size, method):
    """
    Resizes a 3D image to the new size using the specified interpolation method.
    """
    factors = (new_size[0] / image.shape[0], new_size[1] / image.shape[1], new_size[2] / image.shape[2])
    if method == 'bilinear':
        return zoom(image, factors, order=1)
    elif method == 'nearest':
        return zoom(image, factors, order=0)

def process_files(input_folder, output_folder, new_size, task_number):
    """
    Processes and resizes .nii.gz files in the input_folder and saves them to the output_folder.
    """
    for fold, task_folder in enumerate(os.listdir(input_folder)):
        task_path = os.path.join(input_folder, task_folder)
        if not os.path.isdir(task_path):
            continue
        
        # Determine new task folder name
        new_task_number = task_number + fold
        new_task_folder = f"Task{new_task_number}_CSMAbdominalSeg{new_size[0]}Fold{fold + 1}"
        new_task_path = os.path.join(output_folder, new_task_folder)
        
        for root, dirs, files in os.walk(task_path):
            for file in files:
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
                    
                    # Determine the resize method
                    if 'imagesTr' in root or 'imagesTs' in root:
                        resize_method = 'bilinear'
                    elif 'labelsTr' in root or 'labelsTs' in root:
                        resize_method = 'nearest'
                    else:
                        continue
                    
                    # Resize the image
                    resized_data = resize_image(data, new_size, resize_method)
                    
                    # Save the resized image
                    resized_img = nib.Nifti1Image(resized_data, img.affine, img.header)
                    nib.save(resized_img, output_path)
                    print(f"Resized and saved: {output_path}")

# Parameters
input_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out'
output_folder = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/準備data/CSM_dataset/out_'
new_size = 256
task_number = 591
new_size_tuple = (new_size, new_size, 1)
output_folder += str(new_size)

process_files(input_folder, output_folder, new_size_tuple, task_number)
