import os
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

DATAROOT = r"data/sets/nuscenes"
OUT_PATH = r"data/nuscenes_mini_traj_only.npz"

PAST_SECONDS = 2
FUTURE_SECONDS = 6 # not sure

# nuScenes keyframes are at 2 Hz, so:
# 2 seconds past = 4 points
# 6 seconds future = 12 points
EXPECTED_PAST_STEPS = 4
EXPECTED_FUTURE_STEPS = 12


def is_vehicle_category(category_name: str) -> bool:
    return category_name.startswith("vehicle")

def main():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=DATAROOT,
        verbose=True
    )

    helper = PredictHelper(nusc)

    past_list = []
    future_list = []
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

        past_list.append(past_with_current)
        future_list.append(future)

        meta_list.append({
            "instance_token": instance_token,
            "sample_token": sample_token,
            "category_name": category_name
        })

    X = np.stack(past_list)
    y = np.stack(future_list)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        meta=np.array(meta_list, dtype=object)
    )

    print("\nSaved:", OUT_PATH)
    print("X shape:", X.shape)  # num_samples x 5 x 2
    print("y shape:", y.shape)  # num_samples x 12 x 2


if __name__ == "__main__":
    main()