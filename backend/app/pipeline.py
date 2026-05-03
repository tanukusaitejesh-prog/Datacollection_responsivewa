from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class SpatialProcessor:
    def __init__(self, confidence_threshold=0.5, smooth_window=5, poly_order=2):
        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.poly_order = poly_order

        # Keypoint Indices (MediaPipe)
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_SHOULDER = 11
        self.R_SHOULDER = 12

    def process_sequence(self, landmarks, vis_mask):
        """
        landmarks: np.array of shape [T, 33, 3] (X, Y, Z)
        vis_mask: np.array of shape [T, 33] (Visibility scores)
        """
        # 1. Interpolate missing joints (Linear)
        landmarks = self._interpolate(landmarks, vis_mask)

        # 2. Root Centering (Pelvis midpoint)
        root = (landmarks[:, self.L_HIP, :] + landmarks[:, self.R_HIP, :]) / 2.0
        landmarks = landmarks - root[:, np.newaxis, :]

        # 3. Orientation Alignment (XY Plane Rotation)
        landmarks = self._align_orientation(landmarks)

        # 4. Scale Normalization (Median Torso Length)
        landmarks = self._normalize_scale(landmarks)

        # 5. Temporal Smoothing (Savitzky-Golay)
        landmarks = self._smooth(landmarks)

        return landmarks

    def _interpolate(self, landmarks, vis_mask):
        landmarks = landmarks.copy()
        landmarks[vis_mask < self.confidence_threshold] = np.nan
        T, num_joints, dims = landmarks.shape
        df = pd.DataFrame(landmarks.reshape(T, -1))
        df = df.interpolate(method="linear", limit_direction="both").fillna(0)
        return df.values.reshape(T, num_joints, dims)

    def _align_orientation(self, landmarks):
        # Calculate average torso vector (Hip to Shoulder)
        root_2d = (landmarks[:, self.L_HIP, :2] + landmarks[:, self.R_HIP, :2]) / 2.0
        shld_2d = (
            landmarks[:, self.L_SHOULDER, :2] + landmarks[:, self.R_SHOULDER, :2]
        ) / 2.0
        torso_vec = np.mean(shld_2d - root_2d, axis=0)

        # Rotate to point Up (-90 degrees in image space)
        d_theta = (-np.pi / 2.0) - np.arctan2(torso_vec[1], torso_vec[0])
        cos_tr, sin_tr = np.cos(d_theta), np.sin(d_theta)
        R_xy = np.array([[cos_tr, -sin_tr], [sin_tr, cos_tr]])
        landmarks[:, :, :2] = np.matmul(landmarks[:, :, :2], R_xy.T)
        return landmarks

    def _normalize_scale(self, landmarks):
        root_3d = (landmarks[:, self.L_HIP, :] + landmarks[:, self.R_HIP, :]) / 2.0
        shld_3d = (
            landmarks[:, self.L_SHOULDER, :] + landmarks[:, self.R_SHOULDER, :]
        ) / 2.0
        torso_len = np.median(np.linalg.norm(shld_3d - root_3d, axis=1))
        return landmarks / torso_len if torso_len > 1e-5 else landmarks

    def _smooth(self, landmarks):
        T = landmarks.shape[0]
        if T > self.smooth_window:
            window = (
                self.smooth_window
                if self.smooth_window % 2 == 1
                else self.smooth_window + 1
            )
            if window > T:
                window = T if T % 2 == 1 else T - 1
            landmarks = savgol_filter(landmarks, window, self.poly_order, axis=0)
        return landmarks


@dataclass(frozen=True)
class PipelineConfig:
    target_fps: float = 60.0
    smoothing_alpha: float = 0.35
    visibility_threshold: float = 0.5


@dataclass(frozen=True)
class PipelineResult:
    keypoints: np.ndarray
    timestamps: np.ndarray
    processing_meta: dict[str, float | int]
    screening_npy: np.ndarray | None = None


def process_for_screening_app(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: [T, 33, 4] (X, Y, Z, Visibility)
    Returns: [T, 33, 3] processed for Screening App
    """
    coords = keypoints[:, :, :3]
    visibility = keypoints[:, :, 3]
    processor = SpatialProcessor()
    final_npy = processor.process_sequence(coords, visibility)
    return final_npy.astype(np.float32)


def _sorted_unique_timestamps(
    keypoints: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_keypoints = keypoints[order]

    unique_ts, unique_indices = np.unique(sorted_ts, return_index=True)
    unique_keypoints = sorted_keypoints[unique_indices]
    return unique_keypoints, unique_ts


def infer_timestamp_scale_to_ms(timestamps: np.ndarray, target_fps: float) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.shape[0] == 0:
        return 1.0

    candidate_scales = np.array([1.0, 1e-3, 1e-6, 1000.0], dtype=np.float64)

    # Prefer candidates that look like wall-clock epoch milliseconds.
    sample_ts = float(finite[0])
    now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
    epoch_window_ms = 24.0 * 60.0 * 60.0 * 1000.0
    epoch_matches: list[float] = []
    for scale in candidate_scales:
        scaled = sample_ts * float(scale)
        if np.isfinite(scaled) and abs(scaled - now_ms) <= epoch_window_ms:
            epoch_matches.append(float(scale))
    if len(epoch_matches) == 1:
        return epoch_matches[0]

    if finite.shape[0] < 2:
        return 1.0

    sorted_ts = np.sort(finite.astype(np.float64, copy=False))
    deltas = np.diff(sorted_ts)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.shape[0] == 0:
        return 1.0

    median_delta = float(np.median(positive_deltas))
    if median_delta <= 0:
        return 1.0

    expected_delta_ms = 1000.0 / max(float(target_fps), 1e-6)
    best_scale = 1.0
    best_score = float("inf")

    for scale in candidate_scales:
        scaled_delta_ms = median_delta * float(scale)
        if not np.isfinite(scaled_delta_ms) or scaled_delta_ms <= 0:
            continue

        score = abs(np.log(scaled_delta_ms / expected_delta_ms))
        if scaled_delta_ms < 0.1 or scaled_delta_ms > 10000.0:
            score += 5.0

        if score < best_score:
            best_score = float(score)
            best_scale = float(scale)

    return best_scale


def resample_keypoints(
    keypoints: np.ndarray, timestamps: np.ndarray, target_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    if keypoints.shape[0] < 2:
        return keypoints.astype(np.float32, copy=True), timestamps.astype(
            np.float64, copy=True
        )

    unique_keypoints, unique_timestamps = _sorted_unique_timestamps(keypoints, timestamps)
    if unique_keypoints.shape[0] < 2:
        return unique_keypoints.astype(np.float32, copy=True), unique_timestamps.astype(
            np.float64, copy=True
        )

    step_ms = 1000.0 / target_fps
    start_ts = float(unique_timestamps[0])
    end_ts = float(unique_timestamps[-1])
    if end_ts <= start_ts:
        return unique_keypoints.astype(np.float32, copy=True), unique_timestamps.astype(
            np.float64, copy=True
        )

    new_timestamps = np.arange(start_ts, end_ts + 1e-6, step_ms, dtype=np.float64)
    out = np.empty((new_timestamps.shape[0], 33, 4), dtype=np.float32)

    for joint_idx in range(33):
        for channel in range(4):
            out[:, joint_idx, channel] = np.interp(
                new_timestamps,
                unique_timestamps,
                unique_keypoints[:, joint_idx, channel]
            ).astype(np.float32)

    return out, new_timestamps


def exponential_smoothing(keypoints: np.ndarray, alpha: float) -> np.ndarray:
    if keypoints.shape[0] < 2:
        return keypoints.astype(np.float32, copy=True)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    smoothed = keypoints.astype(np.float32, copy=True)

    for t in range(1, smoothed.shape[0]):
        smoothed[t, :, :3] = alpha * smoothed[t, :, :3] + (1.0 - alpha) * smoothed[
            t - 1, :, :3
        ]
        smoothed[t, :, 3] = alpha * smoothed[t, :, 3] + (1.0 - alpha) * smoothed[
            t - 1, :, 3
        ]

    return smoothed


def normalize_skeleton(keypoints: np.ndarray) -> np.ndarray:
    normalized = keypoints.astype(np.float32, copy=True)

    hip_center = (normalized[:, 23, :3] + normalized[:, 24, :3]) * 0.5
    shoulder_center = (normalized[:, 11, :3] + normalized[:, 12, :3]) * 0.5
    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=1)
    torso_length = np.maximum(torso_length, 1e-3)

    normalized[:, :, :3] -= hip_center[:, None, :]
    normalized[:, :, :3] /= torso_length[:, None, None]
    return normalized


def interpolate_missing_keypoints(
    keypoints: np.ndarray, visibility_threshold: float
) -> np.ndarray:
    out = keypoints.astype(np.float32, copy=True)
    frame_axis = np.arange(out.shape[0], dtype=np.float32)

    for joint_idx in range(33):
        visibility = out[:, joint_idx, 3]
        valid_indices = np.where(visibility >= visibility_threshold)[0]

        if valid_indices.size == 0:
            out[:, joint_idx, :3] = 0.0
            out[:, joint_idx, 3] = 0.0
            continue

        if valid_indices.size == 1:
            idx = int(valid_indices[0])
            out[:, joint_idx, :3] = out[idx, joint_idx, :3]
            out[:, joint_idx, 3] = out[idx, joint_idx, 3]
            continue

        x_valid = valid_indices.astype(np.float32)
        for dim in range(3):
            y_valid = out[valid_indices, joint_idx, dim]
            out[:, joint_idx, dim] = np.interp(frame_axis, x_valid, y_valid).astype(
                np.float32
            )

        vis_valid = out[valid_indices, joint_idx, 3]
        out[:, joint_idx, 3] = np.interp(frame_axis, x_valid, vis_valid).astype(np.float32)

    return out


def preprocess_pose_capture(
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    config: PipelineConfig
) -> PipelineResult:
    keypoints_np = np.asarray(keypoints, dtype=np.float32)
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    timestamp_scale_to_ms = infer_timestamp_scale_to_ms(timestamps_np, config.target_fps)
    timestamps_ms = timestamps_np * timestamp_scale_to_ms

    resampled_keypoints, resampled_timestamps = resample_keypoints(
        keypoints_np, timestamps_ms, config.target_fps
    )
    smoothed_keypoints = exponential_smoothing(resampled_keypoints, config.smoothing_alpha)
    normalized_keypoints = normalize_skeleton(smoothed_keypoints)
    filled_keypoints = interpolate_missing_keypoints(
        normalized_keypoints, config.visibility_threshold
    )

    # Generate 100% compatible screening .npy (33 landmarks, specialized processing)
    screening_npy = process_for_screening_app(keypoints_np)

    processing_meta: dict[str, float | int] = {
        "frames_in": int(keypoints_np.shape[0]),
        "frames_out": int(filled_keypoints.shape[0]),
        "target_fps": float(config.target_fps),
        "smoothing_alpha": float(config.smoothing_alpha),
        "visibility_threshold": float(config.visibility_threshold),
        "timestamp_scale_to_ms": float(timestamp_scale_to_ms),
    }

    return PipelineResult(
        keypoints=filled_keypoints,
        timestamps=resampled_timestamps,
        processing_meta=processing_meta,
        screening_npy=screening_npy
    )
