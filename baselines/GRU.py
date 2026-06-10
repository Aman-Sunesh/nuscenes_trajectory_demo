import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import NuScenesTrajectoryDataset

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_only.npz")
METHOD_GROUP = "gru"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results", METHOD_GROUP)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots", METHOD_GROUP)

TEST_SIZE = 0.2
RANDOM_STATE = 43

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

dataset = NuScenesTrajectoryDataset(DATA_PATH)


class GRUBaseline(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=1, output_steps=12, dropout=0.1):
        super().__init__()

        self.output_steps = output_steps

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, output_steps * 2)

    def forward(self, x):
        _, h = self.gru(x)
        last_hidden = h[-1]
        out = self.fc(last_hidden)
        out = out.reshape(x.shape[0], self.output_steps, 2)

        return out


def ADE(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.mean().item()


def FDE(pred, true):
    return torch.norm(pred[-1] - true[-1]).item()


def plot_trajectory_example(X, y_true, y_pred, method_name, idx, ade, fde):
    X_np = X.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    current_np = X_np[-1:]
    true_future_with_current = np.vstack([current_np, y_true_np])
    pred_future_with_current = np.vstack([current_np, y_pred_np])

    plt.figure(figsize=(7, 6))

    plt.plot(X_np[:, 0], X_np[:, 1], marker="o", label="Past trajectory")
    plt.plot(true_future_with_current[:, 0], true_future_with_current[:, 1], marker="o", label="True future")
    plt.plot(pred_future_with_current[:, 0], pred_future_with_current[:, 1], marker="x", label="Predicted future")

    plt.scatter(X_np[-1, 0], X_np[-1, 1], marker="s", label="Current position")

    plt.title(f"{method_name} | sample {idx}\nADE={ade:.3f}, FDE={fde:.3f}")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{method_name}_sample_{idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(RESULTS_DIR, f"{method_name}_per_sample.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "ade", "fde"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_summary_results(results):
    csv_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_summary.csv")
    json_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "hidden_dim",
                "num_layers",
                "epochs",
                "lr",
                "dropout",
                "ade",
                "fde",
                "num_samples"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


def plot_metric_comparison(results):
    methods = [r["method"] for r in results]
    ades = [r["ade"] for r in results]
    fdes = [r["fde"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(x - width / 2, ades, width, label="ADE")
    plt.bar(x + width / 2, fdes, width, label="FDE")

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Error")
    plt.title("GRU Baseline Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_gru(hidden_dim=128, num_layers=1, epochs=100, lr=1e-3, batch_size=64, dropout=0.1, num_plots=5):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    method_name = f"gru_hidden_{hidden_dim}_layers_{num_layers}"

    length = len(dataset)

    X_rows = []
    y_rows = []
    indices = []

    for idx in range(length):
        X, y_true = dataset[idx]

        X_rows.append(X.numpy())          
        y_rows.append(y_true.numpy())     
        indices.append(idx)

    X_all = np.array(X_rows)
    y_all = np.array(y_rows)
    indices = np.array(indices)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all,
        y_all,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = GRUBaseline(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_steps=12,
        dropout=dropout
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

        avg_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{method_name} | Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

    model.eval()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    ade_list = []
    fde_list = []
    per_sample_rows = []

    plot_positions = np.linspace(
        0,
        len(X_test) - 1,
        min(num_plots, len(X_test)),
        dtype=int
    )

    for i in range(len(X_test)):
        X_tensor = X_test_tensor[i]
        y_true_tensor = y_test_tensor[i]
        y_pred_single = y_pred_tensor[i]

        ade = ADE(y_pred_single, y_true_tensor)
        fde = FDE(y_pred_single, y_true_tensor)

        ade_list.append(ade)
        fde_list.append(fde)

        original_idx = int(idx_test[i])

        per_sample_rows.append({
            "sample_idx": original_idx,
            "ade": ade,
            "fde": fde
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X_tensor,
                y_true=y_true_tensor,
                y_pred=y_pred_single,
                method_name=method_name,
                idx=original_idx,
                ade=ade,
                fde=fde
            )

    mean_ade = float(np.mean(ade_list))
    mean_fde = float(np.mean(fde_list))

    save_per_sample_results(method_name, per_sample_rows)

    print(f"\nMethod: {method_name}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": method_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test)
    }


def main():
    results = []

    results.append(run_gru(hidden_dim=64, num_layers=1, epochs=100, lr=1e-3))
    results.append(run_gru(hidden_dim=128, num_layers=1, epochs=100, lr=1e-3))
    results.append(run_gru(hidden_dim=256, num_layers=1, epochs=100, lr=1e-3))
    results.append(run_gru(hidden_dim=128, num_layers=2, epochs=100, lr=1e-3, dropout=0.1))
    results.append(run_gru(hidden_dim=256, num_layers=2, epochs=100, lr=1e-3, dropout=0.1))
    results.append(run_gru(hidden_dim=128, num_layers=3, epochs=100, lr=1e-3, dropout=0.1))
    results.append(run_gru(hidden_dim=256, num_layers=3, epochs=100, lr=1e-3, dropout=0.1))

    save_summary_results(results)
    plot_metric_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()