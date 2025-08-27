from nnunet.dataset_conversion.utils import generate_dataset_json

task_start_num = 605
task_name = "CSMAbdominalSegHuOtsu"
dataset_name="ChungShanMedicalAbdominalSeg_HuOtsu"

for i in range(1, 6):
    print(f"processing in: /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/out/Task{task_start_num+i}_{task_name}Fold{i}")
    generate_dataset_json(output_file=f"/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task{task_start_num+i}_{task_name}Fold{i}/dataset.json", imagesTr_dir=f"/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task{task_start_num+i}_{task_name}Fold{i}/imagesTr", imagesTs_dir=f"/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task{task_start_num+i}_{task_name}Fold{i}/imagesTs", modalities=("CT",), labels={0: "Background", 1: "Abdominal muscles", 2: "VAT"}, dataset_name=dataset_name)