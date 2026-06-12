import os
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

DATAROOT = r"data/sets/nuscenes"
OUT_PATH = r"data/nuscenes_mini_traj_motion_yaw.npz"

PAST_SECONDS = 2
FUTURE_SECONDS = 6 

# nuScenes keyframes are at 2 Hz, so:
# 2 seconds past = 4 points
# 6 seconds future = 12 points
EXPECTED_PAST_STEPS = 4
EXPECTED_FUTURE_STEPS = 12

DT = 0.5  # nuScenes keyframes are 2 Hz


def is_vehicle_category(category_name: str) -> bool:
    return category_name.startswith("vehicle")

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_yaw(ann):
    q = Quaternion(ann["rotation"])
    return q.yaw_pitch_roll[0]

def get_annotation_history(nusc, current_ann, num_past_steps):
    """
    Returns annotation sequence:
    oldest past -> ... -> current
    length = num_past_steps + 1
    """
    anns = [current_ann]
    ann = current_ann
    
    for _ in range(num_past_steps):
        prev_token = ann["prev"]

        if prev_token == "":
            return None
        
        ann = nusc.get("sample_annotation", prev_token)
        anns.append(ann)

    anns = anns[::-1]

    return anns

def build_motion_features(past_with_current_xy, ann_sequence):
    """
    past_with_current_xy: [5, 2], already in current agent frame
    ann_sequence: list of 5 annotation dicts, oldest -> current

    Returns:
    [5, 8] = x, y, vx, vy, ax, ay, cos_yaw, sin_yaw
    """
    xy = np.asarray(past_with_current_xy, dtype=np.float32)
    yaws_list = []

    velocity = np.zeros_like(xy)
    velocity[1:] = (xy[1:] - xy[:-1]) / DT
    velocity[0] = velocity[1]

    acceleration = np.zeros_like(xy)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / DT
    acceleration[0] = acceleration[1]

    # Loop through each annotation in the sequence
    for ann in ann_sequence:
        # Get the yaw for the current annotation
        yaw = get_yaw(ann)
        # Add it to yaws list
        yaws_list.append(yaw)

    yaws = np.array(yaws_list, dtype=np.float32)
    current_yaw = yaws[-1]

    relative_yaw = np.array(
        [normalize_angle(yaw - current_yaw) for yaw in yaws],
        dtype=np.float32
    )

    yaw_features = np.stack(
        [np.cos(relative_yaw), np.sin(relative_yaw)],
        axis=1
    ).astype(np.float32)

    features = np.concatenate(
        [xy, velocity, acceleration, yaw_features],
        axis=1
    )

    return features.astype(np.float32)

def main():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=DATAROOT,
        verbose=True
    )

    helper = PredictHelper(nusc)

    X_list = []
    y_list = []
    meta_list = []

    for ann in tqdm(nusc.sample_annotation, desc="Extracting trajectories"):
        category_name = ann["category_name"]

        # Start with vehicles only.
        if not is_vehicle_category(category_name):
            continue

        instance_token = ann["instance_token"]
        sample_token = ann["sample_token"]

        try:
            past = helper.get_past_for_agent(
                instance_token=instance_token,
                sample_token=sample_token,
                seconds=PAST_SECONDS,
                in_agent_frame=True,
                just_xy=True
            )

            future = helper.get_future_for_agent(
                instance_token=instance_token,
                sample_token=sample_token,
                seconds=FUTURE_SECONDS,
                in_agent_frame=True,
                just_xy=True
            )

        except Exception:
            continue

        past = np.asarray(past, dtype=np.float32)
        future = np.asarray(future, dtype=np.float32)

        # Skip samples without full history/future.
        if past.shape != (EXPECTED_PAST_STEPS, 2):
            continue

        if future.shape != (EXPECTED_FUTURE_STEPS, 2):
            continue

        # The past returned by the helper is usually from nearest past to farthest past.
        # Flip it so the model sees time in chronological order:
        # oldest -> newest.
        past = past[::-1]

        # Add current agent position at t=0.
        # Since we use in_agent_frame=True, current position is [0, 0].
        current = np.zeros((1, 2), dtype=np.float32)
        past_with_current = np.vstack([past, current])
        ann_sequence = get_annotation_history(
            nusc=nusc,
            current_ann=ann,
            num_past_steps=EXPECTED_PAST_STEPS
        )

        if ann_sequence is None:
            continue

        X_features = build_motion_features(
            past_with_current_xy=past_with_current,
            ann_sequence=ann_sequence
        )

        X_list.append(X_features)
        y_list.append(future)

        meta_list.append({
            "instance_token": instance_token,
            "sample_token": sample_token,
            "category_name": category_name,
            "feature_type": "xy_v_a_yaw",
            "feature_names": [
                "x", "y",
                "vx", "vy",
                "ax", "ay",
                "cos_yaw", "sin_yaw"
            ]
        })

    X = np.stack(X_list)
    y = np.stack(y_list)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        meta=np.array(meta_list, dtype=object),
        feature_names=np.array([
            "x", "y",
            "vx", "vy",
            "ax", "ay",
            "cos_yaw", "sin_yaw"
        ], dtype=object)
    )

    print("\nSaved:", OUT_PATH)
    print("X shape:", X.shape)  # num_samples x 5 x 8
    print("y shape:", y.shape)  # num_samples x 12 x 2


if __name__ == "__main__":
    main()