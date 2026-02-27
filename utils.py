import math
import numpy as np
import pandas as pd

from constants import PHASE_RADIUS


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_csv(csv_path)
    X, Y, groups = [], [], []

    for match in df["Match_ID"].unique():
        match_data = df[df["Match_ID"] == match].sort_values(by="Phase")

        for i in range(len(match_data) - 1):
            curr = match_data.iloc[i]
            nxt  = match_data.iloc[i + 1]

            X.append([
                int(curr["Phase"]),
                float(curr["White_X"]),
                float(curr["White_Y"]),
                int(curr["White_R"]),
            ])
            Y.append([float(nxt["White_X"]), float(nxt["White_Y"])])
            groups.append(str(match))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), groups


def euclidean_distances(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    diff = y_true - y_pred
    return np.sqrt((diff ** 2).sum(axis=1))


def apply_boundary_clamp(
    pred_x: float, pred_y: float,
    current_x: float, current_y: float,
    current_r: int, next_r: int,
) -> tuple[int, int, bool]:
    # PUBG kuralı: yeni daire tamamen mevcut dairenin içinde kalmalı
    max_d = current_r - next_r
    if max_d <= 0:
        return int(current_x), int(current_y), False

    d = math.hypot(pred_x - current_x, pred_y - current_y)
    if d > max_d:
        ratio  = max_d / d
        pred_x = current_x + (pred_x - current_x) * ratio
        pred_y = current_y + (pred_y - current_y) * ratio
        return int(pred_x), int(pred_y), True

    return int(pred_x), int(pred_y), False
