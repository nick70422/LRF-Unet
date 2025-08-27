import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, padding_mode='reflect')
        #self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, padding_mode='reflect')
        #self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        
        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.conv1x1(x5))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.final_conv(x)
        return x

# 測試ASPP模組
if __name__ == "__main__":
    aspp = ASPP(in_channels=2048, out_channels=256)
    input_tensor = torch.randn(1, 2048, 32, 32)  # 假設輸入特徵圖大小為32x32
    output = aspp(input_tensor)
    print(output.shape)  # 應該輸出torch.Size([1, 256, 32, 32])