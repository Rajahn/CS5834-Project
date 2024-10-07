import pandas as pd
import numpy as np
from datetime import timedelta

from data_loader import NYCTaxiDataset

# 1. 加载数据
file_path = 'yellow_tripdata_2015-01.csv'
data = pd.read_csv(file_path, usecols=['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude'])

# 过滤异常值（经纬度范围约束在纽约市区）
data = data[(data['pickup_longitude'].between(-74.05, -73.75)) & (data['pickup_latitude'].between(40.63, 40.85))]

# 2. 转换 pickup 时间为半小时区间的时间戳
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['time_slot'] = data['tpep_pickup_datetime'].dt.floor('30T')  # 每30分钟一个时间点

# 3. 创建空间网格 (10x20)
lon_min, lon_max = -74.05, -73.75
lat_min, lat_max = 40.63, 40.85
grid_x, grid_y = 10, 20

# 经纬度到网格的映射
data['grid_x'] = ((data['pickup_longitude'] - lon_min) / (lon_max - lon_min) * grid_x).astype(int)
data['grid_y'] = ((data['pickup_latitude'] - lat_min) / (lat_max - lat_min) * grid_y).astype(int)

# 4. 统计每个时空单元的车流量
volume = data.groupby(['time_slot', 'grid_x', 'grid_y']).size().reset_index(name='flow')

# 创建时间和空间维度的索引
time_slots = pd.date_range(data['time_slot'].min(), data['time_slot'].max(), freq='30T')
# 创建四维数组 (时间, 空间X, 空间Y, 通道=1，只有流量信息
volume_data = np.zeros((len(time_slots), grid_x, grid_y, 1))

# 填充流量数据
for row in volume.itertuples():
    t_idx = (row.time_slot - time_slots[0]) // timedelta(minutes=30)
    volume_data[t_idx, row.grid_x, row.grid_y, 0] = row.flow  # 将车流量填充到通道0

# 5. 划分 70%训练，30%测试
split_idx = int(len(time_slots) * 0.7)
volume_train = volume_data[:split_idx]
volume_test = volume_data[split_idx:]

# 保存
np.savez('volume_train.npz', volume=volume_train)
np.savez('volume_test.npz', volume=volume_test)

if __name__ == "__main__":

    train_path = 'volume_train.npz'
    dataset = NYCTaxiDataset(train_path)
    print(dataset.data.shape)
