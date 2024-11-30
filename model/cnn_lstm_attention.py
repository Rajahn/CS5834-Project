import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: [batch_size, seq_len, 1]
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output, attention_weights


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_maps):
        # feature_maps: [batch_size, channels, time, height, width]
        attention_weights = self.attention(feature_maps)
        # attention_weights: [batch_size, 1, time, height, width]
        attended_features = feature_maps * attention_weights
        return attended_features, attention_weights


class CNNLSTMAttention(nn.Module):
    def __init__(self, in_channel, out_channels, input_size, hidden_size, output_size, drop_prob):
        super(CNNLSTMAttention, self).__init__()

        # CNN部分
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                      out_channels=16,
                      kernel_size=(2, 5, 5),
                      stride=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 3, 3),
                      stride=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        # 空间注意力
        self.spatial_attention = SpatialAttention(32)

        # LSTM部分
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=2,
                            dropout=drop_prob)

        # 时序注意力
        self.temporal_attention = TemporalAttention(hidden_size)

        # 计算CNN输出维度
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 6, 50, 100)
            cnn_output = self.convs(sample_input)
            self.cnn_flat_size = cnn_output.numel() // cnn_output.size(0)
            print(f"CNN output shape: {cnn_output.shape}")
            print(f"CNN flattened size: {self.cnn_flat_size}")

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 降维层
        reduced_size = 32
        self.reduce_dim = nn.Sequential(
            nn.Linear(32, reduced_size),
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
        x_cnn = x[:, :-1].view(batch_size, 1, seq_len - 1, 50, 100)
        cnn_feats = self.convs(x_cnn)

        # 应用空间注意力
        cnn_feats, spatial_weights = self.spatial_attention(cnn_feats)

        # 池化和降维
        cnn_feats = self.adaptive_pool(cnn_feats)
        cnn_feats = cnn_feats.view(batch_size, -1)
        cnn_feats = self.reduce_dim(cnn_feats)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 应用时序注意力
        lstm_feats, temporal_weights = self.temporal_attention(lstm_out)

        # 特征融合
        fusion_feats = torch.cat((cnn_feats, lstm_feats), dim=1)

        # 输出层
        x = self.fc(self.dropout(fusion_feats))

        return torch.sigmoid(x)

    def get_attention_weights(self, x):
        """
        获取注意力权重用于可视化
        """
        batch_size, seq_len, _ = x.shape

        # 获取CNN注意力权重
        x_cnn = x[:, :-1].view(batch_size, 1, seq_len - 1, 50, 100)
        cnn_feats = self.convs(x_cnn)
        _, spatial_weights = self.spatial_attention(cnn_feats)

        # 获取LSTM注意力权重
        lstm_out, _ = self.lstm(x)
        _, temporal_weights = self.temporal_attention(lstm_out)

        return {
            'spatial_attention': spatial_weights.detach(),
            'temporal_attention': temporal_weights.detach()
        }


if __name__ == "__main__":
    # 测试代码
    model = CNNLSTMAttention(in_channel=1,
                    out_channels=[16, 32],
                    input_size=5000,
                    hidden_size=64,
                    output_size=5000,
                    drop_prob=0.5)

    x = torch.randn(32, 7, 5000)
    print("\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # 获取注意力权重
    attention_weights = model.get_attention_weights(x)
    print(f"Spatial attention shape: {attention_weights['spatial_attention'].shape}")
    print(f"Temporal attention shape: {attention_weights['temporal_attention'].shape}")

