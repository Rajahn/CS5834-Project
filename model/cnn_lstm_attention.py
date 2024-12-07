import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns

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
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # [batch_size, seq_len]
        # attention_weights: [batch_size, seq_len]
        attended_output = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)
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
        """获取一个批次数据的注意力权重"""
        batch_size, seq_len, _ = x.shape

        # CNN和空间注意力
        x_cnn = x[:, :-1].view(batch_size, 1, seq_len - 1, 50, 100)
        cnn_feats = self.convs(x_cnn)
        _, spatial_weights = self.spatial_attention(cnn_feats)  # 使用forward方法获取注意力

        # LSTM和时间注意力
        lstm_out, _ = self.lstm(x)
        _, temporal_weights = self.temporal_attention(lstm_out)  # temporal_weights: [batch_size, seq_len]

        return {
            'spatial': spatial_weights.detach().cpu().numpy(),
            'temporal': temporal_weights.detach().cpu().numpy()
        }


def plot_attention_weights(model, dataset, sample_idx=0, save_path=None):
    """绘制注意力权重的可视化图"""
    model.eval()
    with torch.no_grad():
        # 获取一个样本
        sample = dataset.data[sample_idx:sample_idx + 1]
        attention_weights = model.get_attention_weights(sample)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制空间注意力
        spatial_weights = attention_weights['spatial'][0, 0]  # (4, 2, 5)
        # 重塑为2D格式
        spatial_weights_2d = spatial_weights.mean(axis=0)  # 对时间维度取平均，得到(2, 5)

        # 使用插值放大到更清晰的尺寸
        from scipy.ndimage import zoom
        zoom_factor = 10
        spatial_weights_zoomed = zoom(spatial_weights_2d, zoom_factor)

        # 绘制热力图
        sns.heatmap(spatial_weights_zoomed,
                    ax=ax1,
                    cmap='YlOrRd',
                    xticklabels=False,
                    yticklabels=False)
        ax1.set_title('Spatial Attention Weights')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')

        # 绘制时间注意力
        temporal_weights = attention_weights['temporal'][0]  # (seq_len,)
        x = np.arange(len(temporal_weights))
        ax2.bar(x, temporal_weights.flatten())  # 确保是1D数组
        ax2.set_title('Temporal Attention Weights')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Attention Weight')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f't-{len(temporal_weights) - i}' for i in x])

        plt.tight_layout()
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def plot_spatial_attention(model, dataset, sample_idx=0, save_path=None):
    """Visualize spatial attention with actual map regions"""
    with torch.no_grad():
        sample = dataset.data[sample_idx:sample_idx + 1]
        attention_weights = model.get_attention_weights(sample)

        # 创建更大的图像显示更多细节
        plt.figure(figsize=(20, 10))

        # 左侧：原始热力图
        plt.subplot(121)
        spatial_weights = attention_weights['spatial'][0, 0].mean(axis=0)
        weights_reshaped = np.zeros((50, 50))  # 恢复到原始网格大小

        # 使用插值将小尺寸的注意力权重映射回原始网格大小
        from scipy.ndimage import zoom
        zoom_factor = (weights_reshaped.shape[0] / spatial_weights.shape[0],
                       weights_reshaped.shape[1] / spatial_weights.shape[1])
        weights_reshaped = zoom(spatial_weights, zoom_factor)

        # 绘制热力图
        sns.heatmap(weights_reshaped,
                    cmap='YlOrRd',
                    xticklabels=False,
                    yticklabels=False)
        plt.title('Spatial Attention Distribution')

        # 右侧：标注关键区域的解释图
        plt.subplot(122)
        sns.heatmap(weights_reshaped,
                    cmap='YlOrRd',
                    xticklabels=False,
                    yticklabels=False)

        # 标注关键区域
        key_areas = {
            'CBD': (25, 25),
            'Times Square': (20, 30),
            'Airport Connection': (35, 15),
            'Transportation Hub': (15, 35)
        }

        for area, (x, y) in key_areas.items():
            plt.annotate(area, (x, y),
                         xytext=(10, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->'))

        plt.title('Key Areas Highlighted')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def plot_temporal_attention(model, dataset, sample_idx=0, save_path=None):
    """Visualize temporal attention with flow patterns"""
    with torch.no_grad():
        sample = dataset.data[sample_idx:sample_idx + 1]
        attention_weights = model.get_attention_weights(sample)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 上图：时间注意力权重
        temporal_weights = attention_weights['temporal'][0].flatten()
        time_steps = np.arange(len(temporal_weights))
        ax1.bar(time_steps, temporal_weights)

        # 添加时间标注
        time_labels = ['t-6 (3h ago)', 't-5', 't-4', 't-3', 't-2', 't-1', 't (current)']
        ax1.set_xticks(time_steps)
        ax1.set_xticklabels(time_labels, rotation=45)
        ax1.set_title('Temporal Attention Weights Distribution')

        # 下图：对应的流量变化
        flows = sample[0].cpu().numpy()
        ax2.plot(time_steps, flows.mean(axis=1), marker='o', label='Average Flow')

        # 标注特殊时间点
        peak_idx = np.argmax(flows.mean(axis=1))
        ax2.annotate(f'Peak Flow',
                     xy=(peak_idx, flows.mean(axis=1)[peak_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->'))

        ax2.set_xticks(time_steps)
        ax2.set_xticklabels(time_labels, rotation=45)
        ax2.set_title('Corresponding Flow Pattern')
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
# 在训练循环中添加周期性的注意力可视化
def visualize_attention_during_training(model, dataset, epoch, save_dir='attention_vis'):
    """定期保存注意力权重的可视化"""
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    if hasattr(model, 'get_attention_weights'):  # 只对有注意力机制的模型进行可视化
        save_path = f'{save_dir}/attention_epoch_{epoch}.png'
        plot_attention_weights(model, dataset, save_path=save_path)


if __name__ == "__main__":

    pass
