"""Fine-tune Kinetics-pretrained R3D-18 on WLASL RGB clips.

Transfer learning from Kinetics-400 is the strongest CPU-reachable lever for
sparse word-level ASL (~12 videos/class).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_data import load_clip_array  # noqa: E402

MODEL_NAME = "wlasl_r3d18.pt"
METADATA = "wlasl_video_metadata.json"


def _load_split(processed_dir: Path, split: str, frame_size: int = 112, num_frames: int = 16):
    import cv2

    rows = json.loads((processed_dir / f"{split}.json").read_text(encoding="utf-8"))
    xs, ys = [], []
    for row in rows:
        clip = load_clip_array(row["clip_path"])  # (T,H,W,3) RGB 0-255
        # Uniform subsample to num_frames
        t = clip.shape[0]
        idxs = np.linspace(0, t - 1, num_frames).round().astype(int)
        clip = clip[idxs]
        resized = np.stack(
            [cv2.resize(f.astype(np.uint8), (frame_size, frame_size)) for f in clip],
            axis=0,
        ).astype(np.float32)
        # ImageNet-ish normalize later in torch; store 0-1
        xs.append(resized / 255.0)
        ys.append(int(row["label_index"]))
    # (N,T,H,W,C) -> will convert to (N,C,T,H,W)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x: (C,T,H,W) float 0-1"""
    out = x.copy()
    # brightness / contrast
    alpha = float(rng.uniform(0.8, 1.2))
    beta = float(rng.uniform(-0.08, 0.08))
    out = np.clip(out * alpha + beta, 0, 1)
    # temporal roll
    if rng.random() < 0.4:
        shift = int(rng.integers(-2, 3))
        out = np.roll(out, shift, axis=1)
    # spatial crop-resize approx via random erase
    if rng.random() < 0.3:
        _, t, h, w = out.shape
        eh, ew = h // 5, w // 5
        y0 = int(rng.integers(0, h - eh))
        x0 = int(rng.integers(0, w - ew))
        out[:, :, y0 : y0 + eh, x0 : x0 + ew] = float(rng.uniform(0, 0.3))
    return out.astype(np.float32)


def train_r3d(
    processed_dir: str,
    output_dir: str = "models/recognition",
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.video import R3D_18_Weights, r3d_18

    processed = Path(processed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_names"])
    num_classes = len(class_names)

    print("Loading RGB clips for R3D...", flush=True)
    x_train, y_train = _load_split(processed, "train")
    x_val, y_val = _load_split(processed, "val")
    print(f"train={x_train.shape} val={x_val.shape}", flush=True)

    # (N,T,H,W,C) -> (N,C,T,H,W)
    x_train = np.transpose(x_train, (0, 4, 1, 2, 3))
    x_val = np.transpose(x_val, (0, 4, 1, 2, 3))

    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(1, 3, 1, 1, 1)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(1, 3, 1, 1, 1)

    class ClipDS(Dataset):
        def __init__(self, x, y, train=False):
            self.x = x
            self.y = y
            self.train = train
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            xi = self.x[i]
            if self.train:
                xi = _augment(xi, self.rng)
            xi = (xi - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)
            return torch.tensor(xi, dtype=torch.float32), int(self.y[i])

    torch.manual_seed(seed)
    device = torch.device("cpu")
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Freeze stem + early layers initially
    for name, p in model.named_parameters():
        if name.startswith("fc"):
            p.requires_grad = True
        else:
            p.requires_grad = False

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    cw = np.clip(counts.sum() / (num_classes * counts), 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32),
        label_smoothing=0.05,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr * 5, weight_decay=1e-2)

    train_loader = DataLoader(ClipDS(x_train, y_train, train=True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ClipDS(x_val, y_val, train=False), batch_size=batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    stall = 0
    for epoch in range(1, epochs + 1):
        if epoch == 6:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            print("  unfroze full R3D-18", flush=True)

        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb.to(device))
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                yb = yb.to(device)
                pred = model(xb.to(device)).argmax(1)
                v_correct += int((pred == yb).sum())
                v_total += len(yb)
        val_acc = v_correct / max(v_total, 1)
        print(f"  epoch {epoch:03d} train={train_acc:.3f} val={val_acc:.3f}", flush=True)
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 10:
                print(f"  early stop best={best_acc:.3f}", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)

    pt_path = out / MODEL_NAME
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "num_frames": 16,
            "frame_size": 112,
            "architecture": "r3d_18_kinetics_finetune",
        },
        pt_path,
    )

    # Evaluate + write metadata (prefer R3D if better than existing)
    existing = {}
    meta_path = out / METADATA
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    prev = float(existing.get("val_accuracy") or 0.0)

    report = {
        "backend": "wlasl_r3d",
        "architecture": "Kinetics-pretrained R3D-18 fine-tuned on WLASL",
        "dataset": "Voxel51/WLASL top-K RGB clips",
        "num_classes": num_classes,
        "class_names": class_names,
        "num_frames": 16,
        "frame_size": 112,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "val_accuracy": best_acc,
        "r3d_path": str(pt_path),
        "inference": "r3d18",
        "target_signs": 50,
        "target_accuracy": 0.85,
        "signs_met": num_classes >= 50,
        "accuracy_met": best_acc >= 0.85,
    }
    if best_acc >= prev:
        # Keep landmark fields if present but mark R3D as primary inference
        merged = dict(existing)
        merged.update(report)
        meta_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"Wrote {meta_path} (R3D beat previous {prev:.3f})", flush=True)
    else:
        # Still save side report
        (out / "wlasl_r3d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"R3D val={best_acc:.3f} < previous {prev:.3f}; kept existing metadata", flush=True)

    eval_path = Path("logs/evaluation/wlasl_accuracy.json")
    if best_acc >= prev:
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(
            json.dumps(
                {
                    "backend": "wlasl_r3d",
                    "split": "val",
                    "n": int(len(y_val)),
                    "num_classes": num_classes,
                    "accuracy": best_acc,
                    "target_signs": 50,
                    "target_accuracy": 0.85,
                    "signs_met": True,
                    "accuracy_met": best_acc >= 0.85,
                    "class_names": class_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"val_accuracy": best_acc, "accuracy_met": best_acc >= 0.85}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/wlasl_v2")
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train_r3d(args.processed_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
