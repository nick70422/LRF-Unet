import torch
import torch.nn as nn
import numpy as np
import imageio
import os
import random
from PIL import Image


class LITS_dataset(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_label):
        self.path_raw = path_raw
        self.path_label = path_label
        self.filelist = os.listdir(self.path_label)

    def __len__(self):
        return len(self.filelist)

    def read_data(self, index):
        raw = imageio.imread(os.path.join(self.path_raw, self.filelist[index]))  # 直接返回numpy.ndarray 对象
        # raw  = raw .astype(np.float32)
        # raw  = raw  / 200
        raw = torch.from_numpy(raw).float() / 255.

        label = imageio.imread(os.path.join(self.path_label, self.filelist[index]))
        label = torch.from_numpy(label).long()
        # print('read_raw: ' + str(raw.shape) + '|' + 'label' + str(label.shape))
        return raw, label

    def __getitem__(self, index):
        raw, label = self.read_data(index)
        raw = raw.unsqueeze(0)

        # label = label.unsqueeze(0)
        # print(raw.shape, label.shape) torch.Size([4,1,256,256]),torch.Size([4,256,256])
        return raw, label

class CSM_dataset(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_label):
        self.path_raw = path_raw
        self.path_label = path_label
        self.filelist = [f for f in os.listdir(self.path_label) if f.endswith('.png')]

    def __len__(self):
        return len(self.filelist)

    def read_data(self, index):
        # 讀取 .npy 文件作為原始數據
        raw_filename = self.filelist[index].replace('.png', '.npy')
        raw = np.load(os.path.join(self.path_raw, raw_filename))
        raw = torch.from_numpy(raw).float()

        # 讀取 .png 文件作為標籤
        label = Image.open(os.path.join(self.path_label, self.filelist[index]))
        label = torch.from_numpy(np.array(label)).long()

        name_dict = {'name': self.filelist[index].split('.')[0]}
        return raw, label, name_dict

    def __getitem__(self, index):
        raw, label, name_dict = self.read_data(index)
        raw = raw.unsqueeze(0)  # 添加通道維度

        return raw, label, name_dict

def make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, batch=4, n_workers=1):
    dataset_tr = CSM_dataset(tr_path_raw, tr_path_label)
    dataset_ts = CSM_dataset(ts_path_raw, ts_path_label)
    loader_tr = torch.utils.data.DataLoader(dataset=dataset_tr, batch_size=batch, shuffle=False, num_workers=n_workers)
    loader_ts = torch.utils.data.DataLoader(dataset=dataset_ts, batch_size=batch, shuffle=False, num_workers=n_workers)
    dataloaders = {'train': loader_tr, 'val': loader_ts}
    return dataloaders

def make_dataloader(path_raw, path_label, batch=4, n_workers=1):
    dataset = CSM_dataset(path_raw, path_label)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=n_workers)
    return loader

# 测试代码
def main():
    tr_path_raw = r'D:\Nick\medical\code\SAR-U-Net-liver-segmentation-master\preprocessed_data\fold1\imagesTr'
    tr_path_label = r'D:\Nick\medical\code\SAR-U-Net-liver-segmentation-master\preprocessed_data\fold1\labelsTr'
    ts_path_raw = r'D:\Nick\medical\code\SAR-U-Net-liver-segmentation-master\preprocessed_data\fold1\imagesVal'
    ts_path_label = r'D:\Nick\medical\code\SAR-U-Net-liver-segmentation-master\preprocessed_data\fold1\labelsVal'

    dataloader = make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label)
    dataloader = dataloader['val']

    cnt = 0
    for raw, label, name_dict in dataloader:
        # print(raw)
        # print(label)
        print('raw: ' + str(raw.shape))
        print('label: ' + str(label.shape))
        cnt += 1
        if cnt == 5:
            return


if __name__ == '__main__':
    main()
