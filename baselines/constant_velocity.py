import os
import sys
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dataset import NuScenesTrajectoryDataset

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nuscenes_mini_traj_only.npz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

dataset = NuScenesTrajectoryDataset(DATA_PATH)

def plot_trajectory_example(X, y_true, y_pred, method_name, idx, ade, fde):
    X_np = X.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # y_true and y_pred start AFTER the current position.
    # Add current position so the plotted future connects to the observed past.
    current_np = X_np[-1:]
    true_future_with_current = np.vstack([current_np, y_true_np])
    pred_future_with_current = np.vstack([current_np, y_pred_np])
    plt.figure(figsize=(7, 6))

    plt.plot(X_np[:, 0], X_np[:, 1], marker="o", label="Past trajectory")
    plt.plot(true_future_with_current[:, 0], true_future_with_current[:, 1], marker="o", label="True future")
    plt.plot(pred_future_with_current[:, 0], pred_future_with_current[:, 1], marker="x", label="Predicted future")

    plt.scatter(X_np[-1, 0], X_np[-1, 1], marker="s", label="Current position")

    plt.title(f"Constant Velocity ({method_name}) | sample {idx}\nADE={ade:.3f}, FDE={fde:.3f}")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        PLOTS_DIR,
        f"constant_velocity_{method_name}_sample_{idx}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(
        RESULTS_DIR,
        f"constant_velocity_{method_name}_per_sample.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "ade", "fde"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_summary_results(results):
    csv_path = os.path.join(RESULTS_DIR, "constant_velocity_summary.csv")
    json_path = os.path.join(RESULTS_DIR, "constant_velocity_summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "ade", "fde", "num_samples"]
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

    plt.figure(figsize=(8, 6))

    plt.bar(x - width / 2, ades, width, label="ADE")
    plt.bar(x + width / 2, fdes, width, label="FDE")

    plt.xticks(x, methods)
    plt.ylabel("Error")
    plt.title("Constant Velocity Baseline Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "constant_velocity_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

def ADE(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.mean().item()

def FDE(pred, true):
    return torch.norm(pred[-1] - true[-1]).item()

def constant_velocity(type="last_step", num_plots=5):
    length = len(dataset)

    ade_list = []
    fde_list = []
    per_sample_rows = []

    plot_indices = np.linspace(
        0,
        length - 1,
        min(num_plots, length),
        dtype=int
    )

    for idx in range(length):
        X, y_true = dataset[idx]

        current_pos = X[-1]
        future_steps = y_true.shape[0]
        steps = torch.arange(1, future_steps + 1).reshape(-1, 1)

        if type == "last_step":
            prev_pos = X[-2]
            displacement = current_pos - prev_pos

        elif type == "smoothed":
            displacements = X[1:] - X[:-1]
            displacement = displacements.mean(dim=0)

        else:
            raise ValueError(f"Unknown constant velocity type: {type}")

        y_pred = current_pos + displacement * steps

        ade = ADE(y_pred, y_true)
        fde = FDE(y_pred, y_true)

        ade_list.append(ade)
        fde_list.append(fde)

        per_sample_rows.append({
            "sample_idx": idx,
            "ade": ade,
            "fde": fde
        })

        if idx in plot_indices:
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

    save_per_sample_results(type, per_sample_rows)

    print(f"\nConstant Velocity Type: {type}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": f"constant_velocity_{type}",
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": length
    }


def main():
    results = []

    results.append(constant_velocity(type="last_step"))
    results.append(constant_velocity(type="smoothed"))

    save_summary_results(results)
    plot_metric_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()


# Constant velocity is strong for stationary and smooth motion,
# but it fails on nonlinear or maneuvering trajectories.