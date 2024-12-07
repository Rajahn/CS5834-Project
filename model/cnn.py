# model/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_size, output_size, drop_prob):
        super(SimpleCNN, self).__init__()

        self.convs = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算CNN输出维度
        with torch.no_grad():
            x = torch.randn(1, 1, 50, 100)
            cnn_output = self.convs(x)
            self.cnn_flat_size = cnn_output.numel()
            print(f"CNN output shape: {cnn_output.shape}")
            print(f"CNN flattened size: {self.cnn_flat_size}")

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_flat_size, 512),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 5000)
        """
        batch_size, seq_len, _ = x.shape

        # 只使用最后一个时间步的数据
        x = x[:, -1].view(batch_size, 1, 50, 100)
        x = self.convs(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return torch.sigmoid(x)