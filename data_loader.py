import torch
import numpy as np


class NYCTaxiDataset():
    def __init__(self, data_path, window_size=7) -> None:
        self.window_size = window_size
        self.data = torch.from_numpy(self.loading(data_path)).float()

    def loading(self, data_path):
        data = np.load(data_path)['volume']  # shape: (time, 50, 50, 2, 3)
        # 我们只使用flow数据来预测，忽略day_of_week和hour特征
        flow_data = data[:, :, :, :, 0]  # 提取flow值，shape: (time, 50, 50, 2)

        self.max_val, self.min_val = np.max(flow_data), np.min(flow_data)

        # 重塑数据为 (time, 5000) - 将50x50x2展平
        flow_data = flow_data.reshape(flow_data.shape[0], -1)

        # 创建滑动窗口序列
        dataset = []
        for i in range(len(flow_data) - self.window_size + 1):
            dataset.append(flow_data[i:i + self.window_size])

        dataset = np.array(dataset)  # shape: (samples, window_size, 5000)

        # 归一化
        dataset = (dataset - self.min_val) / (self.max_val - self.min_val)
        return dataset

    def denormalize(self, x):
        return x * (self.max_val - self.min_val) + self.min_val

def slidingWindow(seqs, size):

    result = []
    for i in range(seqs.shape[0] - size + 1):
        result.append(seqs[i:i + size, :, :, :])  # (7, 10, 20, 2)
    return result

if __name__ == "__main__":
    train_path = 'data/volume_train.npz'
    dataset = NYCTaxiDataset(train_path)
    print(dataset.data.shape)
    test_path = 'data/volume_test.npz'
    dataset = NYCTaxiDataset(test_path)
    print(dataset.data[0])
