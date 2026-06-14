import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "nuscenes_mini_traj_motion_yaw_map_agents.npz"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results_motion_yaw_map_agents",
    "lstm_topk_conf",
    "lstm_top6_conf_hidden_256_layers_2_checkpoint.pt"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "rolling_online_eval"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_STEPS = 12
NUM_EXAMPLES = 30
USE_TEST_ONLY = True


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim=25,
        hidden_dim=256,
        num_layers=2,
        output_steps=12,
        K=6,
        dropout=0.1
    ):
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


def normalize_with_saved_stats(X, feature_mean, feature_std):
    return (X - feature_mean) / feature_std


def agent_to_global(points_local, anchor_xy, anchor_yaw, mode):
    if mode == "nuscenes_helper":
        points_local = np.stack(
            [
                points_local[:, 1],
                -points_local[:, 0]
            ],
            axis=1
        )    
    
    c = np.cos(anchor_yaw)
    s = np.sin(anchor_yaw)

    R = np.array(
        [
            [c, -s],
            [s,  c]
        ],
        dtype=np.float32
    )

    if mode == "standard":
        return points_local @ R.T + anchor_xy

    if mode == "nuscenes_helper":
        return points_local @ R.T + anchor_xy

    if mode == "transpose":
        return points_local @ R + anchor_xy

    raise ValueError(f"Unknown local-to-global mode: {mode}")


def choose_local_to_global_mode(
    y_raw,
    future_global_xy,
    anchor_global_xy,
    anchor_yaws,
    max_checks=200
):
    n = min(len(y_raw), max_checks)

    standard_errors = []
    transpose_errors = []
    nuscenes_helper_errors = []

    for i in range(n):
        std_global = agent_to_global(
            y_raw[i],
            anchor_global_xy[i],
            anchor_yaws[i],
            mode="standard"
        )

        trans_global = agent_to_global(
            y_raw[i],
            anchor_global_xy[i],
            anchor_yaws[i],
            mode="transpose"
        )

        helper_global = agent_to_global(
            y_raw[i],
            anchor_global_xy[i],
            anchor_yaws[i],
            mode="nuscenes_helper"
        )

        true_global = future_global_xy[i]

        standard_errors.append(
            np.mean(np.linalg.norm(std_global - true_global, axis=1))
        )

        transpose_errors.append(
            np.mean(np.linalg.norm(trans_global - true_global, axis=1))
        )

        nuscenes_helper_errors.append(
            np.mean(np.linalg.norm(helper_global - true_global, axis=1))
        )

    standard_mean = float(np.mean(standard_errors))
    transpose_mean = float(np.mean(transpose_errors))
    nuscenes_helper_mean = float(np.mean(nuscenes_helper_errors))

    print("Local-to-global check:")
    print("standard mean error:", standard_mean)
    print("transpose mean error:", transpose_mean)
    print("nuscenes_helper mean error:", nuscenes_helper_mean)

    errors = {
        "standard": standard_mean,
        "transpose": transpose_mean,
        "nuscenes_helper": nuscenes_helper_mean
    }

    best_mode = min(errors, key=errors.get)

    print("Using local-to-global mode:", best_mode)
    return best_mode


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_online_example(row, example_idx):
    current_xy = row["current_xy"]
    past_trajectory = row["past_trajectory"]
    true_future = row["true_future"]
    initial_pred = row["initial_prediction"]
    online_pred = row["online_prediction"]

    gt_path = np.vstack([current_xy.reshape(1, 2), true_future])
    initial_path = np.vstack([current_xy.reshape(1, 2), initial_pred])
    online_path = np.vstack([current_xy.reshape(1, 2), online_pred])
    past_path = past_trajectory

    plt.figure(figsize=(8, 7))

    plt.plot(
        past_path[:, 0],
        past_path[:, 1],
        marker="o",
        linewidth=2.5,
        label="Past trajectory"
    )

    plt.plot(
        gt_path[:, 0],
        gt_path[:, 1],
        marker="o",
        linewidth=2.5,
        label="True future"
    )

    plt.plot(
        initial_path[:, 0],
        initial_path[:, 1],
        marker="x",
        linewidth=2.5,
        label="Offline predicted future trajectory"
    )

    plt.plot(
        online_path[:, 0],
        online_path[:, 1],
        marker="x",
        linewidth=2.5,
        label="Online predicted future trajectory"
    )

    plt.scatter(
        current_xy[0],
        current_xy[1],
        marker="s",
        s=90,
        label="Current position"
    )

    all_xy = np.vstack([
        past_path,
        gt_path,
        initial_path,
        online_path
    ])

    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    span = max(x_max - x_min, y_max - y_min, 1.0) * 1.25

    plt.xlim(x_center - span / 2, x_center + span / 2)
    plt.ylim(y_center - span / 2, y_center + span / 2)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("global x position")
    plt.ylabel("global y position")

    plt.title(
        "Online-updated trajectory prediction\n"
        f"initial ADE={row['initial_ade']:.3f}, "
        f"online ADE={row['online_ade']:.3f}, "
        f"turn={row['turn_angle_deg']:.1f}°"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        f"online_updated_example_{example_idx:02d}.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_turn_angle_deg(path_xy):
    """
    Compute total direction change along a trajectory.
    Larger value means more turning/curvature.
    """
    deltas = path_xy[1:] - path_xy[:-1]

    speeds = np.linalg.norm(deltas, axis=1)

    valid = speeds > 1e-6
    deltas = deltas[valid]

    if len(deltas) < 2:
        return 0.0

    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    heading_changes = np.diff(headings)

    heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi

    total_turn = np.sum(np.abs(heading_changes))

    return float(np.degrees(total_turn))

def summarize_horizon_errors(point_rows):
    horizon_rows = []

    for step in range(1, OUTPUT_STEPS + 1):
        rows = [
            row for row in point_rows
            if row["target_step"] == step
        ]

        if len(rows) == 0:
            continue

        offline_errors = np.array(
            [row["initial_error"] for row in rows],
            dtype=np.float32
        )

        online_errors = np.array(
            [row["online_error"] for row in rows],
            dtype=np.float32
        )

        horizon_rows.append({
            "target_step": step,
            "mean_offline_error": float(np.mean(offline_errors)),
            "mean_online_error": float(np.mean(online_errors)),
            "mean_error_improvement": float(np.mean(offline_errors - online_errors)),
            "median_offline_error": float(np.median(offline_errors)),
            "median_online_error": float(np.median(online_errors)),
            "num_points": len(rows)
        })

    return horizon_rows


def plot_error_comparison(horizon_rows, summary):
    steps = np.array(
        [row["target_step"] for row in horizon_rows],
        dtype=np.int32
    )

    offline_errors = np.array(
        [row["mean_offline_error"] for row in horizon_rows],
        dtype=np.float32
    )

    online_errors = np.array(
        [row["mean_online_error"] for row in horizon_rows],
        dtype=np.float32
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        steps,
        offline_errors,
        marker="o",
        linewidth=2.5,
        label="Offline prediction error"
    )

    plt.plot(
        steps,
        online_errors,
        marker="o",
        linewidth=2.5,
        label="Online-updated prediction error"
    )

    plt.xlabel("Prediction horizon step")
    plt.ylabel("Mean displacement error")

    plt.title(
        "Offline vs online-updated prediction error\n"
        f"offline ADE={summary['mean_initial_ade']:.3f}, "
        f"online ADE={summary['mean_online_ade']:.3f}, "
        f"ADE improvement={summary['mean_ade_improvement']:.3f}\n"
        f"offline FDE={summary['mean_initial_fde']:.3f}, "
        f"online FDE={summary['mean_online_fde']:.3f}, "
        f"FDE improvement={summary['mean_fde_improvement']:.3f}"
    )

    plt.xticks(steps)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "online_updated_error_by_horizon.png"
    )

    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path

def main():
    data = np.load(DATA_PATH, allow_pickle=True)

    required_fields = [
        "X",
        "y",
        "scene_tokens",
        "instance_tokens",
        "anchor_timestamps",
        "anchor_global_xy",
        "anchor_yaws",
        "future_timestamps",
        "future_global_xy"
    ]

    for field in required_fields:
        if field not in data.files:
            raise KeyError(
                f"Missing {field} in {DATA_PATH}. "
                "Rebuild dataset with future_timestamps and future_global_xy."
            )

    X_raw = data["X"]
    y_raw = data["y"]
    scene_tokens = data["scene_tokens"]
    instance_tokens = data["instance_tokens"]
    anchor_timestamps = data["anchor_timestamps"]
    anchor_global_xy = data["anchor_global_xy"]
    anchor_yaws = data["anchor_yaws"]
    future_timestamps = data["future_timestamps"]
    future_global_xy = data["future_global_xy"]

    local_to_global_mode = choose_local_to_global_mode(
        y_raw=y_raw,
        future_global_xy=future_global_xy,
        anchor_global_xy=anchor_global_xy,
        anchor_yaws=anchor_yaws
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]

    X_norm = normalize_with_saved_stats(
        X_raw,
        feature_mean,
        feature_std
    )

    model = LSTMBaseline(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        output_steps=checkpoint["output_steps"],
        K=checkpoint["K"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if USE_TEST_ONLY:
        eval_indices = checkpoint["idx_test"]
    else:
        eval_indices = np.arange(len(X_raw))

    X_eval_tensor = torch.tensor(
        X_norm[eval_indices],
        dtype=torch.float32
    )

    with torch.no_grad():
        preds_eval, logits_eval = model(X_eval_tensor)

    pred_by_anchor = {}

    for local_i, anchor_idx in enumerate(eval_indices):
        anchor_idx = int(anchor_idx)

        preds = preds_eval[local_i]
        logits = logits_eval[local_i]

        probs = torch.softmax(logits, dim=0)
        conf_k = int(torch.argmax(probs).item())
        conf_prob = float(probs[conf_k].item())

        conf_path_local = preds[conf_k].detach().cpu().numpy()

        conf_path_global = agent_to_global(
            conf_path_local,
            anchor_global_xy[anchor_idx],
            anchor_yaws[anchor_idx],
            local_to_global_mode
        )

        pred_by_anchor[anchor_idx] = {
            "conf_k": conf_k,
            "conf_prob": conf_prob,
            "pred_global": conf_path_global,
            "future_timestamps": future_timestamps[anchor_idx]
        }

    groups = {}

    for idx in eval_indices:
        idx = int(idx)

        key = (
            str(scene_tokens[idx]),
            str(instance_tokens[idx])
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(idx)

    for key in groups:
        groups[key] = sorted(
            groups[key],
            key=lambda idx: int(anchor_timestamps[idx])
        )

    window_rows = []
    point_rows = []

    for key, group_indices in groups.items():
        scene_token, instance_token = key

        for start_idx in group_indices:
            if start_idx not in pred_by_anchor:
                continue

            start_time = int(anchor_timestamps[start_idx])
            current_xy = anchor_global_xy[start_idx]

            past_trajectory = agent_to_global(
                X_raw[start_idx][:, :2],
                anchor_global_xy[start_idx],
                anchor_yaws[start_idx],
                local_to_global_mode
            )

            target_times = future_timestamps[start_idx]
            true_future = future_global_xy[start_idx]

            initial_prediction = pred_by_anchor[start_idx]["pred_global"]

            online_prediction = []
            online_source_anchor_indices = []
            online_source_horizon_steps = []

            valid_window = True

            for target_time in target_times:
                target_time = int(target_time)

                latest_anchor_idx = None
                latest_horizon_step = None

                for candidate_idx in group_indices:
                    if candidate_idx not in pred_by_anchor:
                        continue

                    candidate_time = int(anchor_timestamps[candidate_idx])

                    if candidate_time < start_time:
                        continue

                    if candidate_time >= target_time:
                        break

                    matches = np.where(
                        future_timestamps[candidate_idx] == target_time
                    )[0]

                    if len(matches) == 0:
                        continue

                    latest_anchor_idx = candidate_idx
                    latest_horizon_step = int(matches[-1] + 1)

                if latest_anchor_idx is None:
                    valid_window = False
                    break

                online_point = pred_by_anchor[latest_anchor_idx]["pred_global"][
                    latest_horizon_step - 1
                ]

                online_prediction.append(online_point)
                online_source_anchor_indices.append(latest_anchor_idx)
                online_source_horizon_steps.append(latest_horizon_step)

            if not valid_window:
                continue

            online_prediction = np.asarray(
                online_prediction,
                dtype=np.float32
            )

            initial_errors = np.linalg.norm(
                initial_prediction - true_future,
                axis=1
            )

            online_errors = np.linalg.norm(
                online_prediction - true_future,
                axis=1
            )

            initial_ade = float(np.mean(initial_errors))
            online_ade = float(np.mean(online_errors))

            turn_angle_deg = compute_turn_angle_deg(
                np.vstack([
                    current_xy.reshape(1, 2),
                    true_future
                ])
            )

            initial_fde = float(initial_errors[-1])
            online_fde = float(online_errors[-1])

            ade_improvement = initial_ade - online_ade
            fde_improvement = initial_fde - online_fde

            row = {
                "scene_token": scene_token,
                "instance_token": instance_token,
                "start_anchor_idx": int(start_idx),
                "start_time": int(start_time),
                "initial_ade": initial_ade,
                "online_ade": online_ade,
                "ade_improvement": ade_improvement,
                "initial_fde": initial_fde,
                "online_fde": online_fde,
                "fde_improvement": fde_improvement,
                "turn_angle_deg": turn_angle_deg,
                "num_steps": OUTPUT_STEPS,
                "current_xy": current_xy,
                "past_trajectory": past_trajectory,
                "true_future": true_future,
                "initial_prediction": initial_prediction,
                "online_prediction": online_prediction
            }

            window_rows.append(row)

            for step in range(OUTPUT_STEPS):
                point_rows.append({
                    "scene_token": scene_token,
                    "instance_token": instance_token,
                    "start_anchor_idx": int(start_idx),
                    "target_step": step + 1,
                    "target_time": int(target_times[step]),
                    "source_anchor_idx": int(online_source_anchor_indices[step]),
                    "source_horizon_step": int(online_source_horizon_steps[step]),
                    "true_x": float(true_future[step, 0]),
                    "true_y": float(true_future[step, 1]),
                    "initial_pred_x": float(initial_prediction[step, 0]),
                    "initial_pred_y": float(initial_prediction[step, 1]),
                    "online_pred_x": float(online_prediction[step, 0]),
                    "online_pred_y": float(online_prediction[step, 1]),
                    "initial_error": float(initial_errors[step]),
                    "online_error": float(online_errors[step]),
                    "error_improvement": float(initial_errors[step] - online_errors[step])
                })

    summary_csv_rows = []

    for row in window_rows:
        summary_csv_rows.append({
            "scene_token": row["scene_token"],
            "instance_token": row["instance_token"],
            "start_anchor_idx": row["start_anchor_idx"],
            "start_time": row["start_time"],
            "initial_ade": row["initial_ade"],
            "online_ade": row["online_ade"],
            "ade_improvement": row["ade_improvement"],
            "initial_fde": row["initial_fde"],
            "online_fde": row["online_fde"],
            "fde_improvement": row["fde_improvement"],
            "turn_angle_deg": row["turn_angle_deg"],
            "num_steps": row["num_steps"]
        })

    summary_csv_path = os.path.join(
        OUT_DIR,
        "online_updated_window_summary.csv"
    )

    save_csv(
        summary_csv_path,
        summary_csv_rows,
        fieldnames=[
            "scene_token",
            "instance_token",
            "start_anchor_idx",
            "start_time",
            "initial_ade",
            "online_ade",
            "ade_improvement",
            "initial_fde",
            "online_fde",
            "fde_improvement",
            "turn_angle_deg",
            "num_steps"
        ]
    )

    point_csv_path = os.path.join(
        OUT_DIR,
        "online_updated_points.csv"
    )

    save_csv(
        point_csv_path,
        point_rows,
        fieldnames=[
            "scene_token",
            "instance_token",
            "start_anchor_idx",
            "target_step",
            "target_time",
            "source_anchor_idx",
            "source_horizon_step",
            "true_x",
            "true_y",
            "initial_pred_x",
            "initial_pred_y",
            "online_pred_x",
            "online_pred_y",
            "initial_error",
            "online_error",
            "error_improvement"
        ]
    )

    if len(window_rows) == 0:
        raise RuntimeError("No valid online-updated windows found.")

    mean_initial_ade = float(np.mean([r["initial_ade"] for r in window_rows]))
    mean_online_ade = float(np.mean([r["online_ade"] for r in window_rows]))
    mean_initial_fde = float(np.mean([r["initial_fde"] for r in window_rows]))
    mean_online_fde = float(np.mean([r["online_fde"] for r in window_rows]))

    summary = {
        "num_windows": len(window_rows),
        "mean_initial_ade": mean_initial_ade,
        "mean_online_ade": mean_online_ade,
        "mean_ade_improvement": mean_initial_ade - mean_online_ade,
        "mean_initial_fde": mean_initial_fde,
        "mean_online_fde": mean_online_fde,
        "mean_fde_improvement": mean_initial_fde - mean_online_fde,
        "use_test_only": USE_TEST_ONLY
    }

    summary_path = os.path.join(
        OUT_DIR,
        "online_updated_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    horizon_rows = summarize_horizon_errors(point_rows)

    horizon_csv_path = os.path.join(
        OUT_DIR,
        "online_updated_horizon_error_summary.csv"
    )

    save_csv(
        horizon_csv_path,
        horizon_rows,
        fieldnames=[
            "target_step",
            "mean_offline_error",
            "mean_online_error",
            "mean_error_improvement",
            "median_offline_error",
            "median_online_error",
            "num_points"
        ]
    )

    error_plot_path = plot_error_comparison(
        horizon_rows=horizon_rows,
        summary=summary
    )

    turning_windows = [
        r for r in window_rows
        if r["turn_angle_deg"] >= 20.0
        and r["ade_improvement"] > 0
    ]

    turning_windows = sorted(
        turning_windows,
        key=lambda r: (r["ade_improvement"], r["turn_angle_deg"]),
        reverse=True
    )

    print("Turning windows found:", len(turning_windows))

    if len(turning_windows) < NUM_EXAMPLES:
        print("Not enough turning windows. Filling remaining plots with top improved windows.")

        used_ids = set(r["start_anchor_idx"] for r in turning_windows)

        fallback_windows = [
            r for r in window_rows
            if r["ade_improvement"] > 0
            and r["start_anchor_idx"] not in used_ids
        ]

        fallback_windows = sorted(
            fallback_windows,
            key=lambda r: r["ade_improvement"],
            reverse=True
        )

        turning_windows = turning_windows + fallback_windows

    for example_idx, row in enumerate(turning_windows[:NUM_EXAMPLES]):
        plot_online_example(
            row=row,
            example_idx=example_idx
        )

    print("\nSaved online-updated summary to:", summary_csv_path)
    print("Saved online-updated points to:", point_csv_path)
    print("Saved horizon error summary to:", horizon_csv_path)
    print("Saved error comparison plot to:", error_plot_path)
    print("Saved summary JSON to:", summary_path)
    print("Saved plots to:", OUT_DIR)

    print("\nSummary:")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()