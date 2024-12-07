import os

import pandas as pd
import torch
import math
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from model.cnn_lstm_attention import CNNLSTMAttention, visualize_attention_during_training, plot_attention_weights, \
    plot_spatial_attention, plot_temporal_attention
from model.lstm import MyLSTM
from model.cnn_lstm import CNNLSTM
from data_loader import NYCTaxiDataset
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.utils import shuffle
from configuration import args
from utils import nextBatch, mape, drawPlot

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training Loop
def train(model, train_dataset, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for iteration, batch in enumerate(nextBatch(shuffle(train_dataset.data), args.batch_size)):
        try:
            x, y = batch[:, :-1, :], batch[:, -1, :]

            if args.iscuda:
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 添加维度检查
            if iteration == 0:
                print(f"\nBatch shapes:")
                print(f"Input shape: {x.shape}")
                print(f"Target shape: {y.shape}")

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}, Loss: {loss.item():.8f}")

        except RuntimeError as e:
            print(f"Error in iteration {iteration}:")
            print(f"Batch shape: {batch.shape}")
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            raise e

    avg_loss = total_loss / (iteration + 1)
    train_rmse, train_mae, train_mape, _ = evaluate(model, train_dataset, loss_fn)
    return train_rmse, train_mae, train_mape, avg_loss

# Evaluation Loop
def evaluate(model, dataset, loss_fn):
    model.eval()
    y_hats = []
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in nextBatch(dataset.data, batch_size=args.batch_size):
            x, y = batch[:, :-1, :], batch[:, -1, :]
            if args.iscuda:
                x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()
            count += 1
            y_hats.append(y_hat.detach().cpu().numpy())

    y_hats = np.concatenate(y_hats)
    y_true = dataset.data[:, -1, :]
    y_hats = dataset.denormalize(y_hats)
    y_true = dataset.denormalize(y_true).reshape(y_true.size(0), -1)

    rmse = math.sqrt(mse(y_true, y_hats))
    mae_score = mae(y_true, y_hats)
    mape_score = mape(y_true, y_hats)

    avg_test_loss = total_loss / count  # Calculate average test loss

    return rmse, mae_score, mape_score, avg_test_loss


def plot_comparison(all_metrics):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    for model_name, metrics in all_metrics.items():
        ax1.plot(metrics['loss']['train'], label=f'{model_name}-train')
        ax1.plot(metrics['loss']['test'], label=f'{model_name}-test', linestyle='--')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    for model_name, metrics in all_metrics.items():
        ax2.plot(metrics['rmse']['test'], label=model_name)
    ax2.set_title('Test RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()

    for model_name, metrics in all_metrics.items():
        ax3.plot(metrics['mae']['test'], label=model_name)
    ax3.set_title('Test MAE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.legend()

    for model_name, metrics in all_metrics.items():
        ax4.plot(metrics['mape']['test'], label=model_name)
    ax4.set_title('Test MAPE')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MAPE')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.close()


# 新增: 保存最终结果函数
def save_final_results(all_metrics):
    results = []
    for model_name, metrics in all_metrics.items():
        results.append({
            'Model': model_name,
            'Final Test Loss': metrics['loss']['test'][-1],
            'Final Test RMSE': metrics['rmse']['test'][-1],
            'Final Test MAE': metrics['mae']['test'][-1],
            'Final Test MAPE': metrics['mape']['test'][-1],
            'Best Test RMSE': min(metrics['rmse']['test']),
            'Best Test MAE': min(metrics['mae']['test']),
            'Best Test MAPE': min(metrics['mape']['test'])
        })

    df = pd.DataFrame(results)
    df.to_csv('results/model_comparison.csv', index=False)
    print("\nFinal Results:")
    print(df.to_string())

# Plotting the metrics
def plot_metrics(metrics, model_name):
    fname = f"{model_name}_lr{args.lr}_b{args.batch_size}_h{args.hidden_size}_d{args.drop_prob}_metrics.png"
    drawPlot(metrics, fname, ["loss", "rmse", "mae", "mape"])
    print(f"Plot saved as {fname}")

# Model Training and Evaluation Loop
def run_experiment(train_dataset, test_dataset, model_name, compare_mode=False):
    input_size = 5000
    output_size = 5000

    # 初始化模型
    model = {
        "lstm": lambda: MyLSTM(input_size=input_size,
                               hidden_size=args.hidden_size,
                               output_size=output_size,
                               drop_prob=args.drop_prob),
        "cnnlstm": lambda: CNNLSTM(in_channel=1,
                                   out_channels=[64, 128],
                                   input_size=input_size,
                                   hidden_size=args.hidden_size,
                                   output_size=output_size,
                                   drop_prob=args.drop_prob),
        "cnnlstmattn": lambda: CNNLSTMAttention(in_channel=1,
                                                out_channels=[64, 128],
                                                input_size=input_size,
                                                hidden_size=args.hidden_size,
                                                output_size=output_size,
                                                drop_prob=args.drop_prob)
    }[model_name]()

    print(f"Model {model_name} initialized")

    if args.iscuda:
        model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    metrics = {
        'loss': {'train': [], 'test': []},
        'rmse': {'train': [], 'test': []},
        'mae': {'train': [], 'test': []},
        'mape': {'train': [], 'test': []}
    }

    for epoch in range(args.epochs):
        print(f"========= Epoch {epoch + 1} =========")

        # Training
        train_rmse, train_mae, train_mape, train_loss = train(model, train_dataset, loss_fn, optimizer)
        metrics['loss']['train'].append(train_loss)
        metrics['rmse']['train'].append(train_rmse)
        metrics['mae']['train'].append(train_mae)
        metrics['mape']['train'].append(train_mape)

        # Testing
        test_rmse, test_mae, test_mape, test_loss = evaluate(model, test_dataset, loss_fn)
        metrics['loss']['test'].append(test_loss)
        metrics['rmse']['test'].append(test_rmse)
        metrics['mae']['test'].append(test_mae)
        metrics['mape']['test'].append(test_mape)

        print(f"Epoch {epoch + 1}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, "
              f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

        # if epoch % 10 == 0 and model_name == "cnnlstmattn":  # 只对attention模型进行可视化
        #     visualize_attention_during_training(model, train_dataset, epoch)
        if epoch % 10 == 0 and model_name == "cnnlstmattn":
            plot_spatial_attention(model, train_dataset,
                                   save_path=f'attention_vis/spatial_epoch_{epoch}.png')
            plot_temporal_attention(model, train_dataset,
                                    save_path=f'attention_vis/temporal_epoch_{epoch}.png')

    if not compare_mode:
        # 如果不是比较模式，绘制单个模型的指标图
        plot_metrics([metrics['loss']['train'], metrics['loss']['test'],
                      metrics['rmse']['train'], metrics['rmse']['test'],
                      metrics['mae']['train'], metrics['mae']['test'],
                      metrics['mape']['train'], metrics['mape']['test']], model_name)

    # 保存模型
    torch.save(model.state_dict(), f'models/{model_name}_model.pth')

    print("\nGenerating final attention visualization...")
    # for i in range(5):  # 可视化5个不同的样本
    #     plot_attention_weights(model, test_dataset, sample_idx=i,
    #                            save_path=f'attention_vis/final_attention_sample_{i}.png')

    for i in range(5):
        plot_spatial_attention(model, test_dataset, sample_idx=i,
                               save_path=f'attention_vis/final_spatial_sample_{i}.png')
        plot_temporal_attention(model, test_dataset, sample_idx=i,
                                save_path=f'attention_vis/final_temporal_sample_{i}.png')

    return metrics


if __name__ == "__main__":
    # 加载数据集
    train_dataset = NYCTaxiDataset(data_path='data/volume_train.npz')
    test_dataset = NYCTaxiDataset(data_path='data/volume_test.npz')
    print("Dataset loaded")

    if args.model == "comparison":
        # 比较模式：训练所有模型
        all_metrics = {}
        for model_name in ["lstm", "cnnlstm", "cnnlstmattn"]:
            print(f"\nTraining {model_name}...")
            metrics = run_experiment(train_dataset, test_dataset, model_name, compare_mode=True)
            all_metrics[model_name] = metrics

        # 绘制比较图和保存结果
        plot_comparison(all_metrics)
        save_final_results(all_metrics)
    else:
        # 单模型模式：只训练选定的模型
        run_experiment(train_dataset, test_dataset, args.model)
