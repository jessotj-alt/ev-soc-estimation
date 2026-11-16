from pathlib import Path
import time
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRAIN_FILE = "Final_60trips_Training_BMW_Data(2).csv"
TEST_FILE = "Final_10trips_Testing_BMW_Data(1).csv"
OUTPUT_DIR = Path("assessment10_cnn_gru_outputs")
FEATURES = [
    "Battery Voltage [V]",
    "Battery Current [A]",
    "Battery Temperature [°C]",
    "Ambient Temperature [°C]",
]
TARGET = "SoC [%]"
REQUIRED_COLUMNS = ["Time [s]", *FEATURES, TARGET]
SEQUENCE_LENGTH = 20
TRAIN_STRIDE = 10
VALIDATION_TRIPS = 6
BATCH_SIZE = 1024
MAX_EPOCHS = 4
PATIENCE = 2
LEARNING_RATE = 0.001
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare(path, expected_trips, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.name}. Keep this Python file and both CSV files in the same folder."
        )
    data = pd.read_csv(path, encoding="latin1")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path.name}: {missing_columns}")
    original_rows = len(data)
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data["Trip_ID"] = (data["Time [s]"].diff().fillna(0) < 0).cumsum().astype(int)
    trips = int(data["Trip_ID"].nunique())
    if trips != expected_trips:
        raise ValueError(f"Expected {expected_trips} trips in {path.name}; detected {trips}.")
    print(f"{label}: {path.name}")
    print(f"  Original observations: {original_rows:,}")
    print(f"  Observations after missing-value check: {len(data):,}")
    print(f"  Detected driving trips: {trips}")
    print(f"  SOC range: {data[TARGET].min():.2f}% to {data[TARGET].max():.2f}%")
    return data


def make_sequences(data, mean, std, stride):
    xs, ys, trip_ids, sample_indices = [], [], [], []
    for trip_id, trip in data.groupby("Trip_ID", sort=True):
        x = trip[FEATURES].to_numpy(dtype=np.float32)
        y = trip[TARGET].to_numpy(dtype=np.float32)
        x = (x - mean) / std
        if len(x) < SEQUENCE_LENGTH:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(x, SEQUENCE_LENGTH, axis=0)
        windows = windows.transpose(0, 2, 1)[::stride]
        starts = np.arange(0, len(x) - SEQUENCE_LENGTH + 1, stride)
        xs.append(np.ascontiguousarray(windows, dtype=np.float32))
        ys.append(y[starts + SEQUENCE_LENGTH - 1])
        trip_ids.append(np.full(len(starts), int(trip_id), dtype=np.int32))
        sample_indices.append(starts + SEQUENCE_LENGTH - 1)
    if not xs:
        raise ValueError("No sequences could be generated. Check the sequence length and trip data.")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(trip_ids), np.concatenate(sample_indices)


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CNNGRU(nn.Module):
    def __init__(self, input_features=4, conv_channels=32, hidden_size=48, dropout=0.15):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_features, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        sequence_output, _ = self.gru(x)
        return self.regression_head(sequence_output[:, -1, :]).squeeze(1)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, total_count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = loss_fn(prediction, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_count += len(yb)
    return total_loss / total_count


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            total_loss += loss.item() * len(yb)
            total_count += len(yb)
    return total_loss / total_count


def predict(model, x, batch_size=BATCH_SIZE):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.float32)).to(DEVICE)
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs)


def calculate_metrics(actual, predicted):
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "RMSE (%)": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE (%)": float(mean_absolute_error(actual, predicted)),
        "MAX absolute error (%)": float(np.max(np.abs(error))),
        "STD DEV of error (%)": float(np.std(error)),
        "R²": float(r2_score(actual, predicted)),
    }


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, os_cpu_count()))
    print("ASSESSMENT 10 — CNN-GRU ELECTRIC-VEHICLE SOC ESTIMATION")
    print("=" * 76)
    print(f"Execution device: {DEVICE}")
    print(f"Sequence length: {SEQUENCE_LENGTH} observations; training stride: {TRAIN_STRIDE}")

    train_data = load_and_prepare(Path(TRAIN_FILE), 60, "Training dataset")
    test_data = load_and_prepare(Path(TEST_FILE), 10, "Independent testing dataset")

    train_trip_ids = sorted(train_data["Trip_ID"].unique())
    validation_ids = set(train_trip_ids[-VALIDATION_TRIPS:])
    fit_data = train_data[~train_data["Trip_ID"].isin(validation_ids)]
    validation_data = train_data[train_data["Trip_ID"].isin(validation_ids)]

    mean = fit_data[FEATURES].mean().to_numpy(dtype=np.float32)
    std = fit_data[FEATURES].std(ddof=0).to_numpy(dtype=np.float32)
    std[std == 0] = 1.0

    x_fit, y_fit, _, _ = make_sequences(fit_data, mean, std, TRAIN_STRIDE)
    x_val, y_val, _, _ = make_sequences(validation_data, mean, std, TRAIN_STRIDE)
    print(f"\nTraining trips for model selection: {len(train_trip_ids) - VALIDATION_TRIPS}")
    print(f"Validation trips: {VALIDATION_TRIPS} (kept separate by trip)")
    print(f"Training sequences: {len(y_fit):,}; validation sequences: {len(y_val):,}")

    fit_loader = DataLoader(SequenceDataset(x_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = CNNGRU().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_count = 0
    selection_start = time.perf_counter()
    print("\nStarting trip-separated validation and early stopping...")
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, fit_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        history.append({"Epoch": epoch, "Training RMSE (%)": train_rmse, "Validation RMSE (%)": val_rmse})
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | train RMSE={train_rmse:.4f}% | validation RMSE={val_rmse:.4f}% | time={time.perf_counter()-epoch_start:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break
    selection_time = time.perf_counter() - selection_start
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    print(f"\nSelected epoch: {best_epoch}")
    print("Using the best validation checkpoint for the independent test evaluation.")
    final_model = CNNGRU().to(DEVICE)
    final_model.load_state_dict(best_state)
    torch.save(final_model.state_dict(), OUTPUT_DIR / "assessment10_cnn_gru_model.pth")
    with open(OUTPUT_DIR / "feature_scaler.json", "w", encoding="utf-8") as scaler_file:
        json.dump({"features": FEATURES, "mean": mean.tolist(), "std": std.tolist()}, scaler_file, indent=2)
    x_test, y_test, test_trip_ids, test_indices = make_sequences(test_data, mean, std, 1)
    print(f"Independent test sequences: {len(y_test):,}")

    fit_time = selection_time
    prediction_start = time.perf_counter()
    y_pred = predict(final_model, x_test)
    prediction_time = time.perf_counter() - prediction_start
    metric_values = calculate_metrics(y_test, y_pred)
    metric_values["Model fitting time (s)"] = fit_time
    metric_values["Prediction time (s)"] = prediction_time
    metric_values["Total model computation time (s)"] = fit_time + prediction_time

    results = pd.DataFrame([metric_values])
    results.to_csv(OUTPUT_DIR / "assessment10_performance_metrics.csv", index=False)
    predictions = pd.DataFrame({
        "Test trip number": test_trip_ids + 61,
        "Sample index within trip": test_indices,
        "Actual SOC (%)": y_test,
        "Predicted SOC (%)": y_pred,
        "Error (actual - predicted) (percentage points)": y_test - y_pred,
    })
    predictions.to_csv(OUTPUT_DIR / "assessment10_test_predictions.csv", index=False)

    print("\nINDEPENDENT TEST PERFORMANCE — TRIPS 61–70")
    print("=" * 76)
    for key, value in metric_values.items():
        print(f"{key}: {value:.6f}")

    plot_step = max(1, len(y_test) // 14000)
    plt.figure(figsize=(13, 5.5))
    plt.plot(np.arange(len(y_test))[::plot_step], y_test[::plot_step], label="Actual SOC", linewidth=1.0)
    plt.plot(np.arange(len(y_pred))[::plot_step], y_pred[::plot_step], label="Predicted SOC (CNN-GRU)", linewidth=1.0)
    plt.xlabel("Test sequence index (display points thinned for readability)")
    plt.ylabel("SOC (%)")
    plt.title("Actual vs Predicted SOC — Independent BMW i3 Test Trips 61–70")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "assessment10_soc_tracking.png", dpi=220)
    plt.close()

    error = y_test - y_pred
    plt.figure(figsize=(13, 4.8))
    plt.plot(np.arange(len(error))[::plot_step], error[::plot_step], linewidth=0.8)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Test sequence index (display points thinned for readability)")
    plt.ylabel("Error: actual SOC − predicted SOC (percentage points)")
    plt.title("SOC Estimation Error — CNN-GRU, Test Trips 61–70")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "assessment10_soc_error.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    epochs = [row["Epoch"] for row in history]
    plt.plot(epochs, [row["Training RMSE (%)"] for row in history], marker="o", label="Training RMSE")
    plt.plot(epochs, [row["Validation RMSE (%)"] for row in history], marker="o", label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (%)")
    plt.title("CNN-GRU Training and Validation Learning Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "assessment10_learning_curve.png", dpi=220)
    plt.close()

    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump({"model": "CNN-GRU", "seed": SEED, "sequence_length": SEQUENCE_LENGTH,
                   "available_training_trips": 60, "fitting_trips": 54, "validation_trips": "55-60",
                   "testing_trips": "61-70", "best_epoch": best_epoch,
                   "metrics": metric_values, "device": str(DEVICE)}, file, indent=2)

    print(f"\nAll charts, tables, predictions and run metadata saved in: {OUTPUT_DIR.resolve()}")
    print("Required plots: assessment10_soc_tracking.png and assessment10_soc_error.png")


def os_cpu_count():
    import os
    return os.cpu_count() or 4


if __name__ == "__main__":
    main()
