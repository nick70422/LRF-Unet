import nibabel as nib
import numpy as np
import os
import tqdm

if __name__ == '__main__':
    for fold in range(1, 6):
        print(f"processing fold{fold}...")
        base = f"/mnt/d/Nick/medical/code/DL_models_benchmark-master/準備data/mendeley_dataset/out/Task{575 + fold}_AbdominalSegFold{fold}/"
        dest_base = f"/mnt/d/Nick/medical/code/DL_models_benchmark-master/準備data/mendeley_dataset/out/Task{600 + fold}_mendeley3classFold{fold}/"
        dirs = [base + 'imagesTr/', base + 'imagesTs/', base + 'labelsTr/', base + 'labelsTs/']
        dest_dirs = [dest_base + 'imagesTr/', dest_base + 'imagesTs/', dest_base + 'labelsTr/', dest_base + 'labelsTs/']
        for dir, dest_dir in zip(dirs, dest_dirs):
            os.makedirs(dest_dir, exist_ok=True)
            if 'labels' in dir:
                with tqdm.tqdm(total=len(os.listdir(dir))) as pbar:
                    for file in os.listdir(dir):
                        if not '.nii.gz' in file:
                            continue
                        read_img = nib.load(dir + file)
                        img = read_img.get_fdata()
                        # 处理img矩阵
                        im = np.where(img == 2, 2, np.where(img == 6, 1, 0)).astype(np.uint8)
                        new_file = file.replace('AbdominalSeg', 'mendeley3class')
                        nib.save(nib.Nifti1Image(im, read_img.affine, read_img.header), dest_dir + new_file)
                        pbar.update()
            else:
                with tqdm.tqdm(total=len(os.listdir(dir))) as pbar:
                    for file in os.listdir(dir):
                        if not '.nii.gz' in file:
                            continue
                        new_file = file.replace('AbdominalSeg', 'mendeley3class')
                        nib.save(nib.load(dir + file), dest_dir + new_file)
                        pbar.update()
        print(f"fold{fold} done.")