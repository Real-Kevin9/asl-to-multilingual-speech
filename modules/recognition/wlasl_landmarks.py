"""Pose + hand landmark sequences for WLASL word-level recognition.

RGB CNNs overfit hard on ~12 clips/class. Landmark trajectories are much lower
dimensional and transfer better to live webcam (same MediaPipe stack as alphabet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_LANDMARK_DIR = "data/processed/wlasl_landmarks_v3"
DEFAULT_NUM_FRAMES = 32
POSE_DIM = 33 * 3  # x,y,z
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + HAND_DIM + HAND_DIM  # pose + left + right = 225

HAND_MODEL = "models/recognition/hand_landmarker.task"
POSE_MODEL = "models/recognition/pose_landmarker_lite.task"


def _empty_frame() -> np.ndarray:
    return np.zeros(FEAT_DIM, dtype=np.float32)


def _lm_xyz(landmarks, n: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float32)
    if not landmarks:
        return out.reshape(-1)
    for i, lm in enumerate(landmarks[:n]):
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
        out[i, 2] = float(getattr(lm, "z", 0.0))
    return out.reshape(-1)


def normalize_pose_hands(feat: np.ndarray) -> np.ndarray:
    """Centre on mid-hip and scale by shoulder width (pose indices 11/12, 23/24)."""
    out = feat.copy().reshape(-1, FEAT_DIM)
    for t in range(out.shape[0]):
        pose = out[t, :POSE_DIM].reshape(33, 3)
        # shoulders 11, 12 ; hips 23, 24 in MediaPipe pose
        mid_hip = 0.5 * (pose[23] + pose[24])
        shoulder = np.linalg.norm(pose[11] - pose[12]) + 1e-6
        if shoulder < 1e-4 or not np.isfinite(shoulder):
            continue
        frame = out[t].reshape(-1, 3)
        frame = (frame - mid_hip) / shoulder
        out[t] = frame.reshape(-1)
    return out.astype(np.float32)


class LandmarkExtractor:
    """Frame-wise pose + two-hand landmarks via MediaPipe Tasks."""

    def __init__(
        self,
        hand_model: str = HAND_MODEL,
        pose_model: str = POSE_MODEL,
        *,
        video_mode: bool = False,
    ):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_lib
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat

        self._MpImage = MpImage
        self._ImageFormat = ImageFormat
        self._video_mode = bool(video_mode)
        mode = (
            VisionTaskRunningMode.VIDEO
            if self._video_mode
            else VisionTaskRunningMode.IMAGE
        )

        hand_opts = vision.HandLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=hand_model),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=pose_model),
            running_mode=mode,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            num_poses=1,
        )
        self._hands = vision.HandLandmarker.create_from_options(hand_opts)
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._ts_ms = 0

    def close(self) -> None:
        self._hands.close()
        self._pose.close()

    def reset(self) -> None:
        """Prepare for a new clip (IMAGE mode) or advance the video clock (VIDEO mode).

        MediaPipe ``detect_for_video`` rejects timestamps that decrease, so we must
        never rewind ``_ts_ms`` while in VIDEO mode.
        """
        if not self._video_mode:
            self._ts_ms = 0

    def process_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("opencv-python is required")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._MpImage(image_format=self._ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            self._ts_ms += 33
            pose_res = self._pose.detect_for_video(mp_img, self._ts_ms)
            hand_res = self._hands.detect_for_video(mp_img, self._ts_ms)
        else:
            pose_res = self._pose.detect(mp_img)
            hand_res = self._hands.detect(mp_img)

        pose_vec = _empty_frame()[:POSE_DIM]
        if pose_res.pose_landmarks:
            pose_vec = _lm_xyz(pose_res.pose_landmarks[0], 33)

        left = np.zeros(HAND_DIM, dtype=np.float32)
        right = np.zeros(HAND_DIM, dtype=np.float32)
        if hand_res.hand_landmarks and hand_res.handedness:
            for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = handed[0].category_name.lower() if handed else ""
                vec = _lm_xyz(lms, 21)
                if label.startswith("left"):
                    left = vec
                else:
                    right = vec
        return np.concatenate([pose_vec, left, right]).astype(np.float32)


def sample_indices(n_total: int, n_out: int, offset_frac: float = 0.0) -> List[int]:
    if n_total <= 0:
        return [0] * n_out
    if n_total == 1:
        return [0] * n_out
    usable = max(1, n_total - 1)
    start = int(offset_frac * max(0, n_total - usable * 0.85))
    end = min(n_total, start + max(n_out, int(usable * 0.9)))
    if end <= start:
        start, end = 0, n_total
    span = list(range(start, end))
    if len(span) == 1:
        return [span[0]] * n_out
    idxs = np.linspace(0, len(span) - 1, n_out)
    return [span[int(round(i))] for i in idxs]


def extract_video_landmarks(
    video_path: str,
    extractor: LandmarkExtractor,
    num_frames: int = DEFAULT_NUM_FRAMES,
    offset_frac: float = 0.0,
) -> Optional[np.ndarray]:
    if cv2 is None:
        raise ImportError("opencv-python is required")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    extractor.reset()
    idxs = sample_indices(len(frames), num_frames, offset_frac=offset_frac)
    # IMAGE mode: only run MediaPipe on sampled frames (much faster).
    seq = [extractor.process_bgr(frames[i]) for i in idxs]
    arr = normalize_pose_hands(np.stack(seq, axis=0))
    return arr


def build_landmark_cache(
    records: Sequence[dict],
    class_names: Sequence[str],
    out_dir: str = DEFAULT_LANDMARK_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    train_views: int = 4,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Extract landmark sequences; write train/val manifests under ``out_dir``."""
    from modules.recognition.wlasl_data import stratified_split

    out = Path(out_dir)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    extractor = LandmarkExtractor()
    train_recs, val_recs = stratified_split(list(records), val_fraction=val_fraction, seed=seed)

    def _cache(split_name: str, split_recs: Sequence[dict], multi_view: bool) -> List[dict]:
        rows = []
        offsets = (
            [i / max(train_views - 1, 1) for i in range(train_views)]
            if multi_view and train_views > 1
            else [0.0]
        )
        for j, rec in enumerate(split_recs):
            stem = Path(rec["video_path"]).stem
            for view_i, off in enumerate(offsets):
                suffix = f"_v{view_i}" if len(offsets) > 1 else ""
                npy = clips / f"{rec['gloss']}_{stem}{suffix}.npy"
                if not npy.exists():
                    arr = extract_video_landmarks(
                        rec["video_path"],
                        extractor,
                        num_frames=num_frames,
                        offset_frac=off,
                    )
                    if arr is None:
                        print(f"  skip {rec['video_path']}")
                        continue
                    np.save(npy, arr)
                rows.append(
                    {
                        "gloss": rec["gloss"],
                        "label_index": class_names.index(rec["gloss"]),
                        "video_path": rec["video_path"],
                        "clip_path": str(npy),
                        "view": view_i,
                    }
                )
            if (j + 1) % 25 == 0 or j + 1 == len(split_recs):
                print(f"  {split_name} landmarks {j + 1}/{len(split_recs)}")
        return rows

    try:
        train_m = _cache("train", train_recs, multi_view=True)
        val_m = _cache("val", val_recs, multi_view=False)
    finally:
        extractor.close()

    meta = {
        "dataset": "Voxel51/WLASL",
        "feature": "pose+hands_landmarks",
        "feat_dim": FEAT_DIM,
        "num_frames": num_frames,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "train_views": train_views,
        "n_train": len(train_m),
        "n_val": len(val_m),
        "n_train_videos": len(train_recs),
        "n_val_videos": len(val_recs),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "train.json").write_text(json.dumps(train_m, indent=2), encoding="utf-8")
    (out / "val.json").write_text(json.dumps(val_m, indent=2), encoding="utf-8")
    print(f"Wrote {out} train={len(train_m)} val={len(val_m)}")
    return meta


def load_seq(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def frames_bgr_to_landmark_seq(
    frames_bgr: Sequence[np.ndarray],
    extractor: LandmarkExtractor,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> np.ndarray:
    """Live webcam path: landmarks from a recorded BGR buffer."""
    if not frames_bgr:
        return np.zeros((num_frames, FEAT_DIM), dtype=np.float32)
    extractor.reset()
    idxs = sample_indices(len(frames_bgr), num_frames)
    lo, hi = min(idxs), max(idxs)
    cache: Dict[int, np.ndarray] = {}
    for i in range(lo, hi + 1):
        cache[i] = extractor.process_bgr(frames_bgr[i])
    seq = np.stack([cache[i] for i in idxs], axis=0)
    return normalize_pose_hands(seq)
