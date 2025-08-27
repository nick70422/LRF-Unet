import os
import re

def format_time(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}小時{m}分鐘{s}秒"
    elif m > 0:
        return f"{m}分鐘{s}秒"
    else:
        return f"{s}秒"


def main():
    # 原始路徑，反斜線自動轉正斜線
    base_path = "data/nnUNet_trained_models/nnUNet/2d/時間計算"

    # 遍歷模型名稱（nnUNet層）
    for model_name in os.listdir(base_path):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue
        fold_times = []
        # 遍歷fold（Task581~Task585）
        for fold_num in range(1, 6):
            fold_name = f"Task58{fold_num}_CSMAbdominalSegFold{fold_num}"
            fold_path = os.path.join(model_path, fold_name)
            if not os.path.isdir(fold_path):
                continue
            # 取得trainer資料夾名稱（唯一一個）
            trainer_dirs = [d for d in os.listdir(fold_path) if os.path.isdir(os.path.join(fold_path, d))]
            if not trainer_dirs:
                continue
            trainer_path = os.path.join(fold_path, trainer_dirs[0], "fold_0")
            # 取得唯一txt檔案
            txt_files = [f for f in os.listdir(trainer_path) if f.endswith('.txt')]
            if not txt_files:
                continue
            txt_path = os.path.join(trainer_path, txt_files[0])
            # 讀取txt檔案並累加epoch時間
            total_time = 0.0
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "This epoch took" in line:
                        match = re.search(r'This epoch took ([0-9.]+) s', line)
                        if match:
                            total_time += float(match.group(1))
            fold_times.append(total_time)
        # 計算平均
        if fold_times:
            avg_time = sum(fold_times) / len(fold_times)
            print(f"模型: {model_name}, 平均訓練時間: {format_time(avg_time)}")
        else:
            print(f"模型: {model_name}, 沒有找到有效的fold訓練時間")

if __name__ == "__main__":
    main()
