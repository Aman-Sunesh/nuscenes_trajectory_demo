import os
import sys
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import NuScenesTrajectoryDataset

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_motion_yaw_map_agents.npz")
METHOD_GROUP = "linear_regression"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results_motion_yaw_map_agents", METHOD_GROUP)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots_motion_yaw_map_agents", METHOD_GROUP)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
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
            fieldnames=["method", "alpha", "ade", "fde", "num_samples"]
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

def linear_regression(ridge=False, num_plots=5):
    length = len(dataset)

    X_rows = []
    y_rows = []
    indices = []

    for idx in range(length):
        X, y_true = dataset[idx]

        X_rows.append(X.numpy())          # [5, 25]
        y_rows.append(y_true.numpy())     # [12, 2]
        indices.append(idx)

    X_all = np.array(X_rows)
    y_all = np.array(y_rows)
    indices = np.array(indices)

    X_train_full, X_test_seq, y_train_full, y_test_seq, idx_train_full, idx_test = train_test_split(
        X_all,
        y_all,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    X_train_full_flat = X_train_full.reshape(len(X_train_full), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
    y_train_full_flat = y_train_full.reshape(len(y_train_full), -1)

    if not ridge:
        model = LinearRegression()
        method_name = "linear_regression"
        best_alpha = None
        model.fit(X_train_full_flat, y_train_full_flat)

    else:
        method_name = "ridge_regression"
        best_alpha = None
        best_val_ade = float("inf")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full_flat,
            y_train_full_flat,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        for alpha in RIDGE_ALPHAS:
            candidate_model = Ridge(alpha=alpha)
            candidate_model.fit(X_train, y_train)

            val_pred = candidate_model.predict(X_val).reshape(-1, 12, 2)
            val_true = y_val.reshape(-1, 12, 2)

            val_errors = np.linalg.norm(val_pred - val_true, axis=2)
            val_ade = val_errors.mean()

            print(f"Ridge alpha={alpha} | validation ADE={val_ade:.4f}")

            if val_ade < best_val_ade:
                best_val_ade = val_ade
                best_alpha = alpha

        print(f"Best Ridge alpha: {best_alpha}")

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_full_flat, y_train_full_flat)

    y_pred = model.predict(X_test_flat).reshape(-1, 12, 2)

    X_test = X_test_seq
    y_test = y_test_seq

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
        X_tensor = torch.tensor(X_test[i], dtype=torch.float32)
        y_true_tensor = torch.tensor(y_test[i], dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred[i], dtype=torch.float32)

        ade = ADE(y_pred_tensor, y_true_tensor)
        fde = FDE(y_pred_tensor, y_true_tensor)
        h_err = horizon_errors(y_pred_tensor, y_true_tensor)


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
                X=X_tensor,
                y_true=y_true_tensor,
                y_pred=y_pred_tensor,
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
        "alpha": best_alpha,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }
 
def main():
    results = []

    results.append(linear_regression(ridge=False, num_plots=20))
    results.append(linear_regression(ridge=True, num_plots=20))

    save_summary_results(results)
    save_horizon_results(results)

    plot_metric_comparison(results)
    plot_horizon_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()