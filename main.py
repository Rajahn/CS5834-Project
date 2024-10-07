import os
import torch
import math
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
        x, y = batch[:, :-1, :], batch[:, -1, :]
        if args.iscuda:
            x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item():.8f}")

    avg_loss = total_loss / len(train_dataset.data)
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

# Plotting the metrics
def plot_metrics(metrics, model_name):
    fname = f"{model_name}_lr{args.lr}_b{args.batch_size}_h{args.hidden_size}_d{args.drop_prob}_metrics.png"
    drawPlot(metrics, fname, ["loss", "rmse", "mae", "mape"])
    print(f"Plot saved as {fname}")

# Model Training and Evaluation Loop
def run_experiment():
    # Load Dataset
    train_dataset = NYCTaxiDataset(data_path='data/volume_train.npz')
    test_dataset = NYCTaxiDataset(data_path='data/volume_test.npz')
    print("Dataset loaded")

    # Initialize Model
    if args.model == "lstm":
        model = MyLSTM(input_size=200, hidden_size=args.hidden_size, output_size=200, drop_prob=args.drop_prob)
    elif args.model == "cnnlstm":
        model = CNNLSTM(in_channel=1, out_channels=[64, 128], input_size=200, hidden_size=args.hidden_size, output_size=200, drop_prob=args.drop_prob)
    print(f"Model {args.model} initialized")

    # Set up training components
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    if args.iscuda:
        model = model.to(device)

    # Track metrics across epochs
    train_loss_list, test_loss_list = [], []
    train_rmse_list, test_rmse_list = [], []
    train_mae_list, test_mae_list = [], []
    train_mape_list, test_mape_list = [], []

    for epoch in range(args.epochs):
        print(f"========= Epoch {epoch + 1} =========")

        # Training step
        train_rmse, train_mae, train_mape, train_loss = train(model, train_dataset, loss_fn, optimizer)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_mape_list.append(train_mape)

        # Testing step
        test_rmse, test_mae, test_mape, avg_test_loss = evaluate(model, test_dataset, loss_fn)
        test_loss_list.append(avg_test_loss)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        test_mape_list.append(test_mape)

        print(f"Epoch {epoch + 1}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

    # Save metrics plot
    plot_metrics([train_loss_list, test_loss_list, train_rmse_list, test_rmse_list, train_mae_list, test_mae_list, train_mape_list, test_mape_list], args.model)


if __name__ == "__main__":
    run_experiment()
