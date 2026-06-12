import os
import sys
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import NuScenesTrajectoryDataset

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_motion_yaw_map_agents.npz")
METHOD_GROUP = "mlp"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results_motion_yaw_map_agents", METHOD_GROUP)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots_motion_yaw_map_agents", METHOD_GROUP)
TEST_SIZE = 0.2
RANDOM_STATE = 43

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

dataset = NuScenesTrajectoryDataset(DATA_PATH)

OUTPUT_STEPS = 12
DT = 0.5
HORIZON_TIMES = [(i + 1) * DT for i in range(OUTPUT_STEPS)]
HORIZON_FIELDS = [f"err_{str(t).replace('.', 'p')}s" for t in HORIZON_TIMES]


def horizon_errors(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.detach().cpu().numpy().astype(float)

def normalize_input_features(X_train, X_test, eps=1e-6):
    """
    Normalize input features using training-set statistics only.

    Only X is normalized.
    y remains in original x/y coordinate scale so ADE/FDE stay meaningful.
    """
    feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    feature_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)

    feature_std[feature_std < eps] = 1.0

    X_train_norm = (X_train - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    return X_train_norm, X_test_norm, feature_mean, feature_std


def save_horizon_results(results):
    csv_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_horizon_errors.csv")

    rows = []

    for result in results:
        row = {"method": result["method"]}

        for field, value in zip(HORIZON_FIELDS, result["horizon_errors"]):
            row[field] = float(value)

        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method"] + HORIZON_FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_horizon_comparison(results):
    plt.figure(figsize=(9, 6))

    for result in results:
        plt.plot(
            HORIZON_TIMES,
            result["horizon_errors"],
            marker="o",
            label=result["method"]
        )

    plt.xlabel("Prediction horizon (seconds)")
    plt.ylabel("Mean displacement error")
    plt.title(f"{METHOD_GROUP}: Error by Prediction Horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_horizon_errors.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=125, hidden_dim=128, output_dim=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)   # batch_size × input_dim
        out = self.net(x)               # batch_size × 24
        out = out.reshape(x.shape[0], 12, 2)
        return out
    
def plot_trajectory_example(X, y_true, y_pred, method_name, idx, ade, fde):
    X_np = X.detach().cpu().numpy()
    X_xy = X_np[:, :2]
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    current_np = X_xy[-1:]
    true_future_with_current = np.vstack([current_np, y_true_np])
    pred_future_with_current = np.vstack([current_np, y_pred_np])
    plt.figure(figsize=(7, 6))

    plt.plot(X_xy[:, 0], X_xy[:, 1], marker="o", label="Past trajectory")
    plt.plot(true_future_with_current[:, 0], true_future_with_current[:, 1], marker="o", label="True future")
    plt.plot(pred_future_with_current[:, 0], pred_future_with_current[:, 1], marker="x", label="Predicted future")

    plt.scatter(X_xy[-1, 0], X_xy[-1, 1], marker="s", label="Current position")

    plt.title(f"{method_name} | sample {idx}\nADE={ade:.3f}, FDE={fde:.3f}")
    plt.xlabel("x position")
    plt.ylabel("y position")

    all_xy = np.vstack([X_xy, y_true_np, y_pred_np])

    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    span = max(x_max - x_min, y_max - y_min, 1.0)
    span = span * 1.15

    plt.xlim(x_center - span / 2, x_center + span / 2)
    plt.ylim(y_center - span / 2, y_center + span / 2)

    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        PLOTS_DIR,
        f"{method_name}_sample_{idx}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(
        RESULTS_DIR,
        f"{method_name}_per_sample.csv"
    )

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

    summary_rows = [
        {k: v for k, v in r.items() if k != "horizon_errors"}
        for r in results
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "hidden_dim", "epochs", "lr", "ade", "fde", "num_samples"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=4)


def plot_metric_comparison(results):
    methods = [r["method"] for r in results]
    ades = [r["ade"] for r in results]
    fdes = [r["fde"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(x - width / 2, ades, width, label="ADE")
    plt.bar(x + width / 2, fdes, width, label="FDE")

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Error")
    plt.title(f"{METHOD_GROUP} Baseline Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

def ADE(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.mean().item()

def FDE(pred, true):
    return torch.norm(pred[-1] - true[-1]).item()

def MLP(num_plots=5, epochs=100, lr=1e-3, hidden_dim=128):
    method_name = f"mlp_hidden_{hidden_dim}"
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

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

    X_test_raw = X_test.copy()

    X_train, X_test, feature_mean, feature_std = normalize_input_features(
        X_train,
        X_test
    )

    input_dim = X_train.shape[1] * X_train.shape[2]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=24
    )

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred_train = model(X_train_tensor)

        loss = loss_fn(y_pred_train, y_train_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    ade_list = []
    fde_list = []
    horizon_error_rows = []
    per_sample_rows = []

    plot_positions = np.linspace(
        0,
        len(X_test) - 1,
        min(num_plots, len(X_test)),
        dtype=int
    )

    for i in range(len(X_test)):
        X_plot_tensor = torch.tensor(X_test_raw[i], dtype=torch.float32)
        y_true_tensor = y_test_tensor[i]
        y_pred_single = y_pred_tensor[i]

        ade = ADE(y_pred_single, y_true_tensor)
        fde = FDE(y_pred_single, y_true_tensor)
        h_err = horizon_errors(y_pred_single, y_true_tensor)

        ade_list.append(ade)
        fde_list.append(fde)
        horizon_error_rows.append(h_err)

        original_idx = int(idx_test[i])

        per_sample_rows.append({
            "sample_idx": original_idx,
            "ade": ade,
            "fde": fde
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X_plot_tensor,
                y_true=y_true_tensor,
                y_pred=y_pred_single,
                method_name=method_name,
                idx=original_idx,
                ade=ade,
                fde=fde
            )

    mean_ade = float(np.mean(ade_list))
    mean_fde = float(np.mean(fde_list))
    mean_horizon_errors = np.mean(np.stack(horizon_error_rows), axis=0)

    save_per_sample_results(method_name, per_sample_rows)

    print(f"\nMethod: {method_name}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": method_name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }
 
def main():
    results = []

    results.append(MLP(hidden_dim=64, epochs=100, lr=1e-3, num_plots=20))
    results.append(MLP(hidden_dim=128, epochs=100, lr=1e-3, num_plots=20))
    results.append(MLP(hidden_dim=256, epochs=100, lr=1e-3, num_plots=20))

    save_summary_results(results)
    save_horizon_results(results)

    plot_metric_comparison(results)
    plot_horizon_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()