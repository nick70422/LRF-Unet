from nnunet.dataset_conversion.utils import generate_dataset_json
import os
import pandas as pd

if __name__ == '__main__':
    ## 指定資料夾位置
    #folder_path = r"D:\小智\medical\code\DL_models_benchmark-master\data\nnUNet_raw\nnUNet_raw_data\Task575_AbdominalSeg\labelsTs"
#
    ## 獲取目錄中的所有文件
    #files = os.listdir(folder_path)
#
    ## 遍歷每個文件並重新命名
    #for file_name in files:
    #    if file_name.startswith("AbdominalSeg_") and file_name.endswith("_0000.nii.gz"):
    #        # 獲取編號
    #        file_number = file_name.split('_')[1]
    #        # 新的文件名
    #        new_name = f"AbdominalSeg_{file_number}.nii.gz"
    #        # 獲取完整的文件路徑
    #        old_file_path = os.path.join(folder_path, file_name)
    #        new_file_path = os.path.join(folder_path, new_name)
    #        # 重新命名文件w
    #        os.rename(old_file_path, new_file_path)
    #        print(f"Renamed {file_name} to {new_name}")
#
    #print("All files have been renamed.")
    #generate_dataset_json(output_file="D:/Nick/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task576_AbdominalSeg224/dataset.json", imagesTr_dir="D:/Nick/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task576_AbdominalSeg224/imagesTr", imagesTs_dir="D:/Nick/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task576_AbdominalSeg224/imagesTs", modalities=("CT",), labels={0: "Background", 1: "SAT", 2: "VAT", 3: "Organs", 4: "Spine", 5: "Spine muscles", 6: "Abdominal muscles"}, dataset_name="MendeleyAbdominalSeg_224")
    for i in range(1, 6):
        #print(f"processing in: mendeley_data/out/Task{575+i}_AbdominalSegFold{i}")
        #generate_dataset_json(output_file=f"準備data/mendeley_dataset/out/Task{575+i}_AbdominalSegFold{i}/dataset.json", imagesTr_dir=f"準備data/mendeley_dataset/out/Task{575+i}_AbdominalSegFold{i}/imagesTr", imagesTs_dir=f"準備data/mendeley_dataset/out/Task{575+i}_AbdominalSegFold{i}/imagesTs", modalities=("CT",), labels={0: "Background", 1: "SAT", 2: "VAT", 3: "Organs", 4: "Spine", 5: "Spine muscles", 6: "Abdominal muscles"}, dataset_name="MendeleyAbdominalSeg")
        print(f"processing in: mendeley_data/out/Task{600+i}_mendeley3classFold{i}")
        generate_dataset_json(output_file=f"準備data/mendeley_dataset/out/Task{600+i}_mendeley3classFold{i}/dataset.json", imagesTr_dir=f"準備data/mendeley_dataset/out/Task{600+i}_mendeley3classFold{i}/imagesTr", imagesTs_dir=f"準備data/mendeley_dataset/out/Task{600+i}_mendeley3classFold{i}/imagesTs", modalities=("CT",), labels={0: "Background", 1: "Abdominal muscles", 2: "VAT"}, dataset_name="MendeleyAbdominalSeg_3class")
    #df = pd.read_pickle("準備data/AbdominalSeg_001.pkl")
    #pass