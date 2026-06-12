import os
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

DATAROOT = r"data/sets/nuscenes"
OUT_PATH = r"data/nuscenes_mini_traj_motion_yaw_map_agents.npz"

PAST_SECONDS = 2
FUTURE_SECONDS = 6

# nuScenes keyframes are at 2 Hz, so:
# 2 seconds past = 4 points
# 6 seconds future = 12 points
EXPECTED_PAST_STEPS = 4
EXPECTED_FUTURE_STEPS = 12

DT = 0.5  # nuScenes keyframes are 2 Hz

MAX_LANE_SEARCH_RADIUS = 20.0
NEARBY_RADIUS = 30.0

VEHICLE_TYPE_NAMES = [
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.trailer",
    "vehicle.construction",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.emergency"
]


def is_vehicle_category(category_name: str) -> bool:
    return category_name.startswith("vehicle")


def is_nearby_agent_category(category_name: str) -> bool:
    return (
        category_name.startswith("vehicle")
        or category_name.startswith("human.pedestrian")
    )


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_yaw(ann):
    q = Quaternion(ann["rotation"])
    return q.yaw_pitch_roll[0]


def rotate_global_vector_to_agent_frame(vec_global, current_yaw):
    """
    Rotate a 2D global vector into the current agent frame.
    """
    c = np.cos(-current_yaw)
    s = np.sin(-current_yaw)

    x = c * vec_global[0] - s * vec_global[1]
    y = s * vec_global[0] + c * vec_global[1]

    return np.array([x, y], dtype=np.float32)


def global_point_to_agent_frame(point_global, current_translation, current_yaw):
    """
    Convert a 2D global point into the current agent frame.
    """
    delta = (
        np.asarray(point_global, dtype=np.float32)
        - np.asarray(current_translation[:2], dtype=np.float32)
    )
    return rotate_global_vector_to_agent_frame(delta, current_yaw)


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


def get_map_for_annotation(nusc, nusc_maps, ann):
    sample = nusc.get("sample", ann["sample_token"])
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    if location not in nusc_maps:
        nusc_maps[location] = NuScenesMap(
            dataroot=DATAROOT,
            map_name=location
        )

    return nusc_maps[location], location


def build_lane_features(nusc_map, ann):
    """
    Returns lane features in current agent frame.

    Features:
    lane_dir_x, lane_dir_y,
    dist_to_lane,
    lane_align_cos, lane_align_sin,
    has_lane
    """
    current_xy_global = np.array(ann["translation"][:2], dtype=np.float32)
    current_yaw = get_yaw(ann)

    default_features = np.array(
        [1.0, 0.0, MAX_LANE_SEARCH_RADIUS, 1.0, 0.0, 0.0],
        dtype=np.float32
    )

    try:
        lane_token = nusc_map.get_closest_lane(
            x=float(current_xy_global[0]),
            y=float(current_xy_global[1]),
            radius=MAX_LANE_SEARCH_RADIUS
        )
    except Exception:
        return default_features

    if lane_token is None or lane_token == "":
        return default_features

    try:
        lane_dict = nusc_map.discretize_lanes(
            [lane_token],
            resolution_meters=0.5
        )
    except Exception:
        return default_features

    if lane_token not in lane_dict:
        return default_features

    centerline = np.asarray(lane_dict[lane_token], dtype=np.float32)

    if centerline.ndim != 2 or centerline.shape[0] < 2:
        return default_features

    centerline_xy = centerline[:, :2]

    distances = np.linalg.norm(centerline_xy - current_xy_global[None, :], axis=1)
    closest_idx = int(np.argmin(distances))
    dist_to_lane = float(distances[closest_idx])
    dist_to_lane = min(dist_to_lane, MAX_LANE_SEARCH_RADIUS)

    if closest_idx == 0:
        tangent_global = centerline_xy[1] - centerline_xy[0]
    elif closest_idx == len(centerline_xy) - 1:
        tangent_global = centerline_xy[-1] - centerline_xy[-2]
    else:
        tangent_global = centerline_xy[closest_idx + 1] - centerline_xy[closest_idx - 1]

    tangent_norm = np.linalg.norm(tangent_global)

    if tangent_norm < 1e-6:
        return default_features

    lane_dir_global = tangent_global / tangent_norm
    lane_dir_agent = rotate_global_vector_to_agent_frame(
        lane_dir_global,
        current_yaw
    )

    lane_dir_norm = np.linalg.norm(lane_dir_agent)

    if lane_dir_norm < 1e-6:
        return default_features

    lane_dir_agent = lane_dir_agent / lane_dir_norm

    lane_align_angle = np.arctan2(
        lane_dir_agent[1],
        lane_dir_agent[0]
    )

    lane_align_cos = np.cos(lane_align_angle)
    lane_align_sin = np.sin(lane_align_angle)

    features = np.array(
        [
            lane_dir_agent[0],
            lane_dir_agent[1],
            dist_to_lane,
            lane_align_cos,
            lane_align_sin,
            1.0
        ],
        dtype=np.float32
    )

    return features


def estimate_annotation_speed(nusc, ann):
    """
    Estimate speed from previous annotation of the same instance.
    Returns scalar speed in m/s.
    """
    if ann["prev"] == "":
        return 0.0

    try:
        prev_ann = nusc.get("sample_annotation", ann["prev"])
    except Exception:
        return 0.0

    curr_xy = np.array(ann["translation"][:2], dtype=np.float32)
    prev_xy = np.array(prev_ann["translation"][:2], dtype=np.float32)

    speed = np.linalg.norm(curr_xy - prev_xy) / DT
    return float(speed)


def build_nearby_agent_features(nusc, ann):
    """
    Current-frame nearby-agent features.

    Features:
    nearby_agent_count,
    nearest_agent_dist,
    nearest_agent_speed
    """
    sample = nusc.get("sample", ann["sample_token"])
    current_xy = np.array(ann["translation"][:2], dtype=np.float32)

    count = 0
    nearest_dist = NEARBY_RADIUS
    nearest_speed = 0.0

    for other_ann_token in sample["anns"]:
        if other_ann_token == ann["token"]:
            continue

        other_ann = nusc.get("sample_annotation", other_ann_token)
        other_category = other_ann["category_name"]

        if not is_nearby_agent_category(other_category):
            continue

        other_xy = np.array(other_ann["translation"][:2], dtype=np.float32)
        dist = float(np.linalg.norm(other_xy - current_xy))

        if dist > NEARBY_RADIUS:
            continue

        count += 1

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_speed = estimate_annotation_speed(nusc, other_ann)

    features = np.array(
        [
            float(count),
            float(nearest_dist),
            float(nearest_speed)
        ],
        dtype=np.float32
    )

    return features


def build_vehicle_type_features(category_name):
    """
    One-hot vehicle subtype feature.
    """
    one_hot = np.zeros(len(VEHICLE_TYPE_NAMES), dtype=np.float32)

    for i, type_name in enumerate(VEHICLE_TYPE_NAMES):
        if category_name.startswith(type_name):
            one_hot[i] = 1.0
            return one_hot

    return one_hot


def build_context_features(nusc, nusc_maps, ann):
    nusc_map, location = get_map_for_annotation(
        nusc=nusc,
        nusc_maps=nusc_maps,
        ann=ann
    )

    lane_features = build_lane_features(
        nusc_map=nusc_map,
        ann=ann
    )

    nearby_features = build_nearby_agent_features(
        nusc=nusc,
        ann=ann
    )

    vehicle_type_features = build_vehicle_type_features(
        category_name=ann["category_name"]
    )

    context_features = np.concatenate(
        [
            lane_features,
            nearby_features,
            vehicle_type_features
        ],
        axis=0
    ).astype(np.float32)

    return context_features, location


def main():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=DATAROOT,
        verbose=True
    )

    helper = PredictHelper(nusc)

    nusc_maps = {}

    X_list = []
    y_list = []
    meta_list = []

    feature_names = [
        "x", "y",
        "vx", "vy",
        "ax", "ay",
        "cos_yaw", "sin_yaw",

        "lane_dir_x",
        "lane_dir_y",
        "dist_to_lane",
        "lane_align_cos",
        "lane_align_sin",
        "has_lane",

        "nearby_agent_count",
        "nearest_agent_dist",
        "nearest_agent_speed",
    ]

    for type_name in VEHICLE_TYPE_NAMES:
        feature_names.append(f"type_{type_name.replace('.', '_')}")

    for ann in tqdm(nusc.sample_annotation, desc="Extracting trajectories + map/agent context"):
        category_name = ann["category_name"]

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

        if past.shape != (EXPECTED_PAST_STEPS, 2):
            continue

        if future.shape != (EXPECTED_FUTURE_STEPS, 2):
            continue

        past = past[::-1]

        current = np.zeros((1, 2), dtype=np.float32)
        past_with_current = np.vstack([past, current])

        ann_sequence = get_annotation_history(
            nusc=nusc,
            current_ann=ann,
            num_past_steps=EXPECTED_PAST_STEPS
        )

        if ann_sequence is None:
            continue

        motion_features = build_motion_features(
            past_with_current_xy=past_with_current,
            ann_sequence=ann_sequence
        )

        context_features, location = build_context_features(
            nusc=nusc,
            nusc_maps=nusc_maps,
            ann=ann
        )

        context_sequence = np.repeat(
            context_features.reshape(1, -1),
            repeats=motion_features.shape[0],
            axis=0
        )

        X_features = np.concatenate(
            [motion_features, context_sequence],
            axis=1
        ).astype(np.float32)

        X_list.append(X_features)
        y_list.append(future)

        meta_list.append({
            "instance_token": instance_token,
            "sample_token": sample_token,
            "category_name": category_name,
            "location": location,
            "feature_type": "xy_v_a_yaw_lane_nearby_type",
            "feature_names": feature_names
        })

    X = np.stack(X_list)
    y = np.stack(y_list)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        meta=np.array(meta_list, dtype=object),
        feature_names=np.array(feature_names, dtype=object)
    )

    print("\nSaved:", OUT_PATH)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature count:", len(feature_names))
    print("Feature names:", feature_names)


if __name__ == "__main__":
    main()
