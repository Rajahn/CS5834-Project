import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, in_channel, out_channels, input_size, hidden_size, output_size, drop_prob):
        super(CNNLSTM, self).__init__()

        # 进一步减少CNN输出维度
        self.convs = nn.Sequential(
            # 第一层: 减少通道数和特征图大小
            nn.Conv3d(in_channels=in_channel,
                      out_channels=16,  # 减少通道数
                      kernel_size=(2, 5, 5),  # 增大卷积核
                      stride=(1, 2, 2)),  # 使用步长减少特征图大小
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # 第二层: 进一步减少特征
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 3, 3),
                      stride=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # LSTM部分
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=2,
                            dropout=drop_prob)

        # 计算CNN输出特征维度并打印调试信息
        with torch.no_grad():
            # 使用示例输入计算CNN输出大小
            sample_input = torch.randn(1, 1, 6, 50, 100)
            cnn_output = self.convs(sample_input)
            self.cnn_flat_size = cnn_output.numel() // cnn_output.size(0)
            print(f"CNN output shape: {cnn_output.shape}")
            print(f"CNN flattened size: {self.cnn_flat_size}")

        # 使用自适应池化来固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 降维层
        reduced_size = 32  # 固定CNN输出通道数
        self.reduce_dim = nn.Sequential(
            nn.Linear(32, reduced_size),  # CNN最后输出32通道
            nn.ReLU(),
            nn.Dropout(drop_prob)
        )

        # 最终的全连接层
        self.fc = nn.Linear(reduced_size + hidden_size, output_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        """
        x: (batch, seq_len, 5000)
        """
        batch_size, seq_len, _ = x.shape

        # CNN处理
        # 重塑输入为5维张量: (batch, channel, time, height, width)
        x_cnn = x[:, :-1].view(batch_size, 1, seq_len - 1, 50, 100)
        cnn_feats = self.convs(x_cnn)
        #print(f"After conv shape: {cnn_feats.shape}")

        # 使用自适应池化固定输出大小
        cnn_feats = self.adaptive_pool(cnn_feats)
        #print(f"After adaptive pool shape: {cnn_feats.shape}")

        # 展平并降维
        cnn_feats = cnn_feats.view(batch_size, -1)
        cnn_feats = self.reduce_dim(cnn_feats)
        #print(f"After reduction shape: {cnn_feats.shape}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_feats = lstm_out[:, -1]  # 取最后一个时间步
        #print(f"LSTM features shape: {lstm_feats.shape}")

        # 特征融合
        fusion_feats = torch.cat((cnn_feats, lstm_feats), dim=1)
        #print(f"Fusion features shape: {fusion_feats.shape}")

        # 全连接层输出
        x = self.fc(self.dropout(fusion_feats))
        return torch.sigmoid(x)


if __name__ == "__main__":
    # 测试代码
    model = CNNLSTM(in_channel=1,
                    out_channels=[16, 32],
                    input_size=5000,
                    hidden_size=64,
                    output_size=5000,
                    drop_prob=0.5)

    # 创建测试数据
    x = torch.randn(32, 7, 5000)
    print("\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")