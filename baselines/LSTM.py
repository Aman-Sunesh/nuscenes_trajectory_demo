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

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_motion_yaw_map_agents.npz")
METHOD_GROUP = "lstm_topk_conf"
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

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, num_layers=1, output_steps=12, K=6, dropout=0.1):
        super().__init__()

        self.output_steps = output_steps
        self.K = K

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.traj_head = nn.Linear(hidden_dim, self.K * output_steps * 2)
        self.conf_head = nn.Linear(hidden_dim, self.K)

    def forward(self, x):
        _, (h, c) = self.lstm(x)

        last_hidden = h[-1]

        preds = self.traj_head(last_hidden)
        preds = preds.reshape(x.shape[0], self.K, self.output_steps, 2)

        logits = self.conf_head(last_hidden)

        return preds, logits


def ADE(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.mean().item()

def FDE(pred, true):
    return torch.norm(pred[-1] - true[-1]).item()

def topk_wta_conf_loss(preds, logits, true, alpha=0.1):
    """
    preds:  [B, K, T, 2]
    logits: [B, K]
    true:   [B, T, 2]

    Winner-takes-all top-K loss.
    For each sample, choose the trajectory with the lowest ADE,
    train that trajectory against the ground truth, and train the
    confidence head to select the same best trajectory.
    """
    errors = torch.norm(preds - true[:, None, :, :], dim=-1)  # [B, K, T]
    ade_per_k = errors.mean(dim=-1)                           # [B, K]

    best_k = ade_per_k.argmin(dim=1)                           # [B]
    batch_idx = torch.arange(preds.shape[0], device=preds.device)

    best_preds = preds[batch_idx, best_k]                      # [B, T, 2]

    loss_reg = nn.functional.smooth_l1_loss(best_preds, true)
    loss_cls = nn.functional.cross_entropy(logits, best_k)

    loss = loss_reg + alpha * loss_cls

    return loss


def topk_metrics_single(preds, logits, true):
    """
    preds: [K, T, 2]
    logits: [K]
    true:  [T, 2]

    Compute top-K metrics for one sample.
    minADE/minFDE use the best trajectory among K using ground truth.
    confADE/confFDE use the model's highest-confidence trajectory.
    Horizon errors are computed from the best-ADE trajectory.
    """
    errors = torch.norm(preds - true[None, :, :], dim=-1)   # [K, T]

    ade_per_k = errors.mean(dim=-1)                         # [K]
    fde_per_k = errors[:, -1]                               # [K]

    best_ade_k = int(torch.argmin(ade_per_k).item())
    best_fde_k = int(torch.argmin(fde_per_k).item())

    probs = torch.softmax(logits, dim=0)
    conf_k = int(torch.argmax(probs).item())
    conf_prob = float(probs[conf_k].item())

    min_ade = float(ade_per_k[best_ade_k].item())
    min_fde = float(fde_per_k[best_fde_k].item())
    conf_ade = float(ade_per_k[conf_k].item())
    conf_fde = float(fde_per_k[conf_k].item())

    best_pred_by_ade = preds[best_ade_k]
    conf_pred = preds[conf_k]

    h_err = horizon_errors(best_pred_by_ade, true)

    return {
        "min_ade": min_ade,
        "min_fde": min_fde,
        "best_ade_k": best_ade_k,
        "best_fde_k": best_fde_k,
        "conf_ade": conf_ade,
        "conf_fde": conf_fde,
        "conf_k": conf_k,
        "conf_prob": conf_prob,
        "best_pred_by_ade": best_pred_by_ade,
        "conf_pred": conf_pred,
        "horizon_errors": h_err
    }

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

    save_path = os.path.join(PLOTS_DIR, f"{method_name}_sample_{idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(RESULTS_DIR, f"{method_name}_per_sample.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "min_ade",
                "min_fde",
                "best_ade_k",
                "best_fde_k",
                "conf_ade",
                "conf_fde",
                "conf_k",
                "conf_prob"
            ]
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
            fieldnames=[
                "method",
                "K",
                "alpha",
                "hidden_dim",
                "num_layers",
                "epochs",
                "lr",
                "dropout",
                "min_ade",
                "min_fde",
                "conf_ade",
                "conf_fde",
                "num_samples"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=4)


def plot_metric_comparison(results):
    methods = [r["method"] for r in results]
    min_ades = [r["min_ade"] for r in results]
    min_fdes = [r["min_fde"] for r in results]
    conf_ades = [r["conf_ade"] for r in results]
    conf_fdes = [r["conf_fde"] for r in results]

    x = np.arange(len(methods))
    width = 0.2

    plt.figure(figsize=(11, 6))

    plt.bar(x - 1.5 * width, min_ades, width, label="minADE@K")
    plt.bar(x - 0.5 * width, min_fdes, width, label="minFDE@K")
    plt.bar(x + 0.5 * width, conf_ades, width, label="confADE@K")
    plt.bar(x + 1.5 * width, conf_fdes, width, label="confFDE@K")

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Error")
    plt.title("Top-K LSTM with Confidence Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_lstm(hidden_dim=128, num_layers=1, epochs=100, lr=1e-3, batch_size=64, dropout=0.1, K=6, alpha=0.1, num_plots=5):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    method_name = f"lstm_top{K}_conf_hidden_{hidden_dim}_layers_{num_layers}"

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

    input_dim = X_train.shape[-1]

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

    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_steps=12,
        K=K,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds, logits = model(X_batch)
            loss = topk_wta_conf_loss(preds, logits, y_batch, alpha=alpha)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

        avg_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{method_name} | Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

    model.eval()

    with torch.no_grad():
        y_pred_tensor, logits_tensor = model(X_test_tensor)

    ade_list = []
    fde_list = []
    conf_ade_list = []
    conf_fde_list = []
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

        preds_single = y_pred_tensor[i]      
        logits_single = logits_tensor[i]     

        metrics = topk_metrics_single(
            preds_single,
            logits_single,
            y_true_tensor
        )

        ade_list.append(metrics["min_ade"])
        fde_list.append(metrics["min_fde"])
        conf_ade_list.append(metrics["conf_ade"])
        conf_fde_list.append(metrics["conf_fde"])
        horizon_error_rows.append(metrics["horizon_errors"])

        original_idx = int(idx_test[i])

        per_sample_rows.append({
            "sample_idx": original_idx,
            "min_ade": metrics["min_ade"],
            "min_fde": metrics["min_fde"],
            "best_ade_k": metrics["best_ade_k"],
            "best_fde_k": metrics["best_fde_k"],
            "conf_ade": metrics["conf_ade"],
            "conf_fde": metrics["conf_fde"],
            "conf_k": metrics["conf_k"],
            "conf_prob": metrics["conf_prob"]
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X_plot_tensor,
                y_true=y_true_tensor,
                y_pred=metrics["best_pred_by_ade"],
                method_name=method_name,
                idx=original_idx,
                ade=metrics["min_ade"],
                fde=metrics["min_fde"]
            )

    mean_min_ade = float(np.mean(ade_list))
    mean_min_fde = float(np.mean(fde_list))
    mean_conf_ade = float(np.mean(conf_ade_list))
    mean_conf_fde = float(np.mean(conf_fde_list))
    mean_horizon_errors = np.mean(np.stack(horizon_error_rows), axis=0)

    save_per_sample_results(method_name, per_sample_rows)

    checkpoint_path = os.path.join(RESULTS_DIR, f"{method_name}_checkpoint.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_steps": OUTPUT_STEPS,
            "K": K,
            "dropout": dropout,
            "alpha": alpha,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "idx_test": idx_test,
            "method_name": method_name,
        },
        checkpoint_path
    )

    print("Saved checkpoint:", checkpoint_path)

    print(f"\nMethod: {method_name}")
    print(f"minADE@{K}:", mean_min_ade)
    print(f"minFDE@{K}:", mean_min_fde)
    print(f"confADE@{K}:", mean_conf_ade)
    print(f"confFDE@{K}:", mean_conf_fde)

    return {
        "method": method_name,
        "K": K,
        "alpha": alpha,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "min_ade": mean_min_ade,
        "min_fde": mean_min_fde,
        "conf_ade": mean_conf_ade,
        "conf_fde": mean_conf_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }

def main():
    results = []

    results.append(
        run_lstm(
            hidden_dim=64,
            num_layers=1,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=128,
            num_layers=1,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=256,
            num_layers=1,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=128,
            num_layers=2,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=256,
            num_layers=2,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=128,
            num_layers=3,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    results.append(
        run_lstm(
            hidden_dim=256,
            num_layers=3,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    save_summary_results(results)
    save_horizon_results(results)

    plot_metric_comparison(results)
    plot_horizon_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()