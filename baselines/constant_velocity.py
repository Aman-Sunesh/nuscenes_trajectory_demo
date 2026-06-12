import os
import sys
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import NuScenesTrajectoryDataset

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_motion_yaw_map_agents.npz")
METHOD_GROUP = "constant_velocity"
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

    # y_true and y_pred start AFTER the current position.
    # Add current position so the plotted future connects to the observed past.
    current_np = X_xy[-1:]
    true_future_with_current = np.vstack([current_np, y_true_np])
    pred_future_with_current = np.vstack([current_np, y_pred_np])
    plt.figure(figsize=(7, 6))

    plt.plot(X_xy[:, 0], X_xy[:, 1], marker="o", label="Past trajectory")
    plt.plot(true_future_with_current[:, 0], true_future_with_current[:, 1], marker="o", label="True future")
    plt.plot(pred_future_with_current[:, 0], pred_future_with_current[:, 1], marker="x", label="Predicted future")

    plt.scatter(X_xy[-1, 0], X_xy[-1, 1], marker="s", label="Current position")

    plt.title(f"Constant Velocity ({method_name}) | sample {idx}\nADE={ade:.3f}, FDE={fde:.3f}")
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
        f"{METHOD_GROUP}_{method_name}_sample_{idx}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(
        RESULTS_DIR,
        f"{METHOD_GROUP}_{method_name}_per_sample.csv"
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
            fieldnames=["method", "ade", "fde", "num_samples"]
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
    plt.title("Constant Velocity Baseline Comparison")
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

def constant_velocity(type="last_step", num_plots=5):
    total_length = len(dataset)

    all_indices = np.arange(total_length)

    _, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    ade_list = []
    fde_list = []
    per_sample_rows = []

    plot_positions = np.linspace(
        0,
        len(test_indices) - 1,
        min(num_plots, len(test_indices)),
        dtype=int
    )

    horizon_error_rows = []

    for i, idx in enumerate(test_indices):
        idx = int(idx)
        X, y_true = dataset[idx]

        X_xy = X[:, :2]

        current_pos = X_xy[-1]
        future_steps = y_true.shape[0]
        steps = torch.arange(1, future_steps + 1, dtype=torch.float32).reshape(-1, 1)

        if type == "last_step":
            prev_pos = X_xy[-2]
            displacement = current_pos - prev_pos

        elif type == "smoothed":
            displacements = X_xy[1:] - X_xy[:-1]
            displacement = displacements.mean(dim=0)

        else:
            raise ValueError(f"Unknown constant velocity type: {type}")

        y_pred = current_pos + displacement * steps

        ade = ADE(y_pred, y_true)
        fde = FDE(y_pred, y_true)
        h_err = horizon_errors(y_pred, y_true)

        ade_list.append(ade)
        fde_list.append(fde)
        horizon_error_rows.append(h_err)

        per_sample_rows.append({
            "sample_idx": idx,
            "ade": ade,
            "fde": fde
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                method_name=type,
                idx=idx,
                ade=ade,
                fde=fde
            )

    mean_ade = float(np.mean(ade_list))
    mean_fde = float(np.mean(fde_list))
    mean_horizon_errors = np.mean(np.stack(horizon_error_rows), axis=0)

    save_per_sample_results(type, per_sample_rows)

    print(f"\nConstant Velocity Type: {type}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": f"constant_velocity_{type}",
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(test_indices),
        "horizon_errors": mean_horizon_errors
    }


def main():
    results = []

    results.append(constant_velocity(type="last_step", num_plots=20))
    results.append(constant_velocity(type="smoothed", num_plots=20))

    save_summary_results(results)
    save_horizon_results(results)

    plot_metric_comparison(results)
    plot_horizon_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()


# Constant velocity is strong for stationary and smooth motion,
# but it fails on nonlinear or maneuvering trajectories.