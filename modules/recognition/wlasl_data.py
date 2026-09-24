"""WLASL word-level sign data: download, frame extraction, train/val split.

Uses the Voxel51 mirror of WLASL on Hugging Face (academic C-UDA licence).
Selects the top-K most frequent glosses so the proposal target of ≥50 signs is
met with the densest available clips.

Preprocessing improvements for sparse data:
* crop to the signer bounding box when available (matches live torso crop)
* multiple temporal windows per training video
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HF_DATASET_ID = "Voxel51/WLASL"
DEFAULT_RAW_DIR = "data/raw/wlasl"
DEFAULT_PROCESSED_DIR = "data/processed/wlasl_v2"
DEFAULT_NUM_CLASSES = 50
DEFAULT_FRAMES = 24
DEFAULT_FRAME_SIZE = 160
DEFAULT_TRAIN_VIEWS = 4


def load_samples_index(samples_json: Path) -> List[dict]:
    data = json.loads(Path(samples_json).read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"Unexpected samples.json layout in {samples_json}")
    return samples


def gloss_counts(samples: Sequence[dict]) -> Counter:
    return Counter(_gloss_label(s) for s in samples if _gloss_label(s))


def select_top_glosses(samples: Sequence[dict], num_classes: int = DEFAULT_NUM_CLASSES) -> List[str]:
    counts = gloss_counts(samples)
    if len(counts) < num_classes:
        raise ValueError(
            f"Only {len(counts)} glosses available; cannot select {num_classes}."
        )
    return [g for g, _ in counts.most_common(num_classes)]


def _gloss_label(sample: dict) -> Optional[str]:
    gloss = sample.get("gloss")
    if isinstance(gloss, dict):
        label = gloss.get("label")
        return str(label).strip().lower() if label else None
    if isinstance(gloss, str):
        return gloss.strip().lower()
    return None


def _bbox_xyxy(sample: dict) -> Optional[Tuple[float, float, float, float]]:
    """Return normalised xyxy bbox if present, else None."""
    box = sample.get("bounding_box") or {}
    detections = box.get("detections") if isinstance(box, dict) else None
    if not detections:
        return None
    det = detections[0]
    bb = det.get("bounding_box") if isinstance(det, dict) else None
    if not bb or len(bb) != 4:
        return None
    x, y, w, h = [float(v) for v in bb]
    return x, y, x + w, y + h


def filter_samples(samples: Sequence[dict], glosses: Sequence[str]) -> List[dict]:
    wanted = {g.lower() for g in glosses}
    out = []
    for sample in samples:
        label = _gloss_label(sample)
        path = sample.get("filepath")
        if label in wanted and path:
            out.append({
                "gloss": label,
                "filepath": path,
                "metadata": sample.get("metadata") or {},
                "bbox": _bbox_xyxy(sample),
            })
    return out


def download_samples_json(raw_dir: str = DEFAULT_RAW_DIR) -> Path:
    from huggingface_hub import hf_hub_download

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        filename="samples.json",
    )
    dest = raw / "samples.json"
    if not dest.exists():
        dest.write_bytes(Path(cached).read_bytes())
    return dest


def download_videos(
    records: Sequence[dict],
    raw_dir: str = DEFAULT_RAW_DIR,
    max_workers: int = 4,
) -> List[dict]:
    """Download selected mp4 files; returns records with local ``video_path``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import hf_hub_download

    raw = Path(raw_dir)
    videos_root = raw / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    def _one(rec: dict) -> dict:
        rel = rec["filepath"]
        local = videos_root / Path(rel).name
        if not local.exists() or local.stat().st_size == 0:
            cached = hf_hub_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                filename=rel,
            )
            local.write_bytes(Path(cached).read_bytes())
        out = dict(rec)
        out["video_path"] = str(local)
        return out

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, rec) for rec in records]
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(futures):
                print(f"  downloaded {i}/{len(futures)} clips")
    return results


def sample_frame_indices(
    n_frames_total: int,
    n_out: int,
    offset_frac: float = 0.0,
) -> List[int]:
    """Uniform sample of ``n_out`` indices, optionally shifted within the clip."""
    if n_frames_total <= 0:
        return [0] * n_out
    if n_frames_total == 1:
        return [0] * n_out
    if n_frames_total < n_out:
        idxs = list(range(n_frames_total))
        while len(idxs) < n_out:
            idxs.extend(range(n_frames_total))
        return idxs[:n_out]

    usable = n_frames_total
    # Leave a small margin so offset windows stay inside the clip.
    margin = max(0, usable - n_out)
    start = int(round(offset_frac * margin))
    start = max(0, min(start, margin))
    end = start + max(n_out - 1, 1)
    end = min(end, usable - 1)
    return list(np.linspace(start, end, n_out, dtype=int))


def crop_signer_rgb(
    rgb: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    pad: float = 0.15,
    center_fallback: float = 0.75,
) -> np.ndarray:
    """Crop around the signer. Falls back to a centred torso crop for live webcam."""
    h, w = rgb.shape[:2]
    if bbox is None:
        side = min(h, w) * center_fallback
        cx, cy = w / 2.0, h * 0.45
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(w, cx + side / 2))
        y2 = int(min(h, cy + side / 2))
    else:
        x1n, y1n, x2n, y2n = bbox
        bw = max(1e-3, x2n - x1n)
        bh = max(1e-3, y2n - y1n)
        x1n = max(0.0, x1n - pad * bw)
        y1n = max(0.0, y1n - pad * bh)
        x2n = min(1.0, x2n + pad * bw)
        y2n = min(1.0, y2n + pad * bh)
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return rgb
    return rgb[y1:y2, x1:x2]


def extract_clip_frames(
    video_path: str,
    num_frames: int = DEFAULT_FRAMES,
    frame_size: int = DEFAULT_FRAME_SIZE,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    offset_frac: float = 0.0,
    use_center_crop: bool = True,
) -> Optional[np.ndarray]:
    """Return RGB float32 array shaped ``(T, H, W, 3)`` in 0–255, or None."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices(max(total, 1), num_frames, offset_frac=offset_frac)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if bbox is not None or use_center_crop:
            rgb = crop_signer_rgb(rgb, bbox=bbox)
        rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if len(frames) != num_frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


def frames_bgr_to_clip(
    frames_bgr: Sequence[np.ndarray],
    num_frames: int = DEFAULT_FRAMES,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> np.ndarray:
    """Convert a live BGR buffer into the same crop/resize pipeline as training."""
    import cv2

    idxs = sample_frame_indices(len(frames_bgr), num_frames)
    out = []
    for i in idxs:
        rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        rgb = crop_signer_rgb(rgb, bbox=None)
        rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        out.append(rgb.astype(np.float32))
    return np.stack(out, axis=0)


def stratified_split(
    records: Sequence[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    by_gloss: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_gloss[rec["gloss"]].append(rec)

    rng = random.Random(seed)
    train, val = [], []
    for gloss, items in sorted(by_gloss.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        n_val = max(1, int(round(len(items) * val_fraction)))
        n_val = min(n_val, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def prepare_wlasl(
    num_classes: int = DEFAULT_NUM_CLASSES,
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    num_frames: int = DEFAULT_FRAMES,
    frame_size: int = DEFAULT_FRAME_SIZE,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_workers: int = 4,
    skip_download: bool = False,
    train_views: int = DEFAULT_TRAIN_VIEWS,
) -> dict:
    """Download top-K WLASL clips and write train/val manifests + frame cache."""
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    cache_dir = processed / "clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples_json = download_samples_json(raw_dir=raw_dir)
    samples = load_samples_index(samples_json)
    class_names = select_top_glosses(samples, num_classes=num_classes)
    selected = filter_samples(samples, class_names)
    print(f"Selected {len(class_names)} glosses, {len(selected)} videos")

    if skip_download:
        videos_root = raw / "videos"
        records = []
        for rec in selected:
            local = videos_root / Path(rec["filepath"]).name
            if local.exists():
                item = dict(rec)
                item["video_path"] = str(local)
                records.append(item)
    else:
        records = download_videos(selected, raw_dir=raw_dir, max_workers=max_workers)

    train_recs, val_recs = stratified_split(records, val_fraction=val_fraction, seed=seed)

    def _cache_split(
        split_name: str,
        split_recs: Sequence[dict],
        multi_view: bool,
    ) -> List[dict]:
        out = []
        offsets = (
            [i / max(train_views - 1, 1) for i in range(train_views)]
            if multi_view and train_views > 1
            else [0.0]
        )
        total_jobs = len(split_recs) * len(offsets)
        done = 0
        for rec in split_recs:
            stem = Path(rec["video_path"]).stem
            bbox = rec.get("bbox")
            for view_i, offset in enumerate(offsets):
                suffix = f"_v{view_i}" if multi_view and train_views > 1 else ""
                npy_path = cache_dir / f"{rec['gloss']}_{stem}{suffix}.npy"
                if not npy_path.exists():
                    arr = extract_clip_frames(
                        rec["video_path"],
                        num_frames=num_frames,
                        frame_size=frame_size,
                        bbox=bbox,
                        offset_frac=offset,
                        use_center_crop=True,
                    )
                    if arr is None:
                        print(f"  skip unreadable {rec['video_path']}")
                        done += 1
                        continue
                    np.save(npy_path, arr)
                out.append(
                    {
                        "gloss": rec["gloss"],
                        "label_index": class_names.index(rec["gloss"]),
                        "video_path": rec["video_path"],
                        "clip_path": str(npy_path),
                        "view": view_i,
                    }
                )
                done += 1
                if done % 50 == 0 or done == total_jobs:
                    print(f"  {split_name} frames {done}/{total_jobs}")
        return out

    print(f"Extracting train frames ({train_views} temporal views)...")
    train_manifest = _cache_split("train", train_recs, multi_view=True)
    print("Extracting val frames (1 view)...")
    val_manifest = _cache_split("val", val_recs, multi_view=False)

    meta = {
        "dataset": HF_DATASET_ID,
        "licence": "C-UDA (academic / non-commercial)",
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_frames": num_frames,
        "frame_size": frame_size,
        "train_views": train_views,
        "signer_crop": True,
        "n_train": len(train_manifest),
        "n_val": len(val_manifest),
        "n_train_videos": len(train_recs),
        "n_val_videos": len(val_recs),
        "gloss_counts": dict(Counter(r["gloss"] for r in records)),
    }
    (processed / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (processed / "train.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    (processed / "val.json").write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifests under {processed} (train clips={len(train_manifest)})")
    return meta


def load_clip_array(clip_path: str) -> np.ndarray:
    return np.load(clip_path).astype(np.float32)


def iter_manifest(processed_dir: str, split: str) -> List[dict]:
    path = Path(processed_dir) / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))
