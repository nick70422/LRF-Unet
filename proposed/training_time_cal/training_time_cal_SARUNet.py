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
    # 只處理SAR-U-Net_log.txt
    log_path = os.path.join(os.path.dirname(__file__), 'SAR-U-Net_log.txt')
    if not os.path.exists(log_path):
        print("SAR-U-Net_log.txt 檔案不存在")
        return
    fold_times = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fold_time = 0.0
    epoch_count = 0
    for line in lines:
        if line.startswith('Processing fold'):
            if epoch_count == 50:
                fold_times.append(fold_time)
            fold_time = 0.0
            epoch_count = 0
        elif 'Time taken:' in line:
            match = re.search(r'Time taken: ([0-9.]+) seconds', line)
            if match:
                fold_time += float(match.group(1))
                epoch_count += 1
    # 最後一個fold
    if epoch_count == 50:
        fold_times.append(fold_time)
    if fold_times:
        avg_time = sum(fold_times) / len(fold_times)
        print(f"SAR-U-Net 平均訓練時間: {format_time(avg_time)}")
    else:
        print("SAR-U-Net 沒有找到有效的fold訓練時間")

if __name__ == "__main__":
    main()
