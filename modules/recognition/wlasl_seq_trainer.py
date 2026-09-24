"""Train a strong WLASL classifier on pose+hand landmark sequences.

Primary model: small Transformer encoder (PyTorch, CPU-friendly).
Secondary: HistGradientBoosting on temporal stats (ensemble / fallback).
Target: push toward the proposal ≥85% holdout accuracy on ≥50 glosses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from modules.recognition.wlasl_landmarks import (
    DEFAULT_LANDMARK_DIR,
    DEFAULT_NUM_FRAMES,
    FEAT_DIM,
    load_seq,
)

MODEL_PT = "wlasl_landmark_transformer.pt"
MODEL_SKLEARN = "wlasl_landmark_hgb.joblib"
METADATA = "wlasl_video_metadata.json"


def _load_split(processed_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = json.loads((processed_dir / f"{split}.json").read_text(encoding="utf-8"))
    xs, ys = [], []
    for row in rows:
        xs.append(load_seq(row["clip_path"]))
        ys.append(int(row["label_index"]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def augment_seq(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy temporal / geometric noise on one (T, D) sequence."""
    out = x.astype(np.float32, copy=True)
    t, d = out.shape
    out += rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    if rng.random() < 0.4:
        n_drop = int(rng.integers(1, max(2, d // 15)))
        cols = rng.choice(d, size=n_drop, replace=False)
        out[:, cols] = 0.0
    if rng.random() < 0.45 and t > 4:
        # Cheap temporal shift/crop instead of full interp warp
        shift = int(rng.integers(-max(1, t // 8), max(2, t // 8)))
        out = np.roll(out, shift, axis=0)
        if shift > 0:
            out[:shift] = 0.0
        elif shift < 0:
            out[shift:] = 0.0
    if rng.random() < 0.35:
        drop = max(1, t // 8)
        start = int(rng.integers(0, t - drop + 1))
        out[start : start + drop] = 0.0
    return out


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized-ish batch aug (still per-sample, but less Python in hot path)."""
    return np.stack([augment_seq(x[i], rng) for i in range(len(x))], axis=0)


def temporal_stats(x: np.ndarray) -> np.ndarray:
    """Pool a (T, D) sequence into a fixed vector for tree models."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    first = x[0]
    last = x[-1]
    # Velocities
    vel = np.diff(x, axis=0)
    vmean = vel.mean(axis=0) if len(vel) else np.zeros_like(mean)
    vstd = vel.std(axis=0) if len(vel) else np.zeros_like(mean)
    return np.concatenate([mean, std, mn, mx, first, last, vmean, vstd]).astype(np.float32)


def _build_torch_model(num_classes: int, feat_dim: int, d_model: int = 128, nhead: int = 4, layers: int = 3):
    import torch
    from torch import nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 128):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(feat_dim, d_model)
            self.pe = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.25,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, num_classes),
            )

        def forward(self, x):
            # x: (B, T, D)
            h = self.proj(x)
            h = self.pe(h)
            h = self.encoder(h)
            h = self.norm(h.mean(dim=1))
            return self.head(h)

    return Model()


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    *,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    aug_copies: int = 6,
    seed: int = 42,
    patience: int = 15,
    d_model: int = 128,
    nhead: int = 4,
    layers: int = 3,
    online_aug: bool = False,
) -> Tuple[object, dict]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    class _OnlineSeq(Dataset):
        """Augment on the fly so Colab does not store N full copies in RAM."""

        def __init__(self, x: np.ndarray, y: np.ndarray, repeats: int, base_seed: int):
            self.x = x
            self.y = y
            self.repeats = max(1, repeats)
            self.base_seed = base_seed

        def __len__(self) -> int:
            return len(self.y) * self.repeats

        def __getitem__(self, index: int):
            n = len(self.y)
            src = index % n
            row = self.x[src]
            if index >= n:
                row = augment_seq(row, np.random.default_rng(self.base_seed + index))
            return torch.from_numpy(np.ascontiguousarray(row)), int(self.y[src])

    if online_aug:
        repeats = max(1, aug_copies + 1)
        print(
            f"  online aug (repeats={repeats}, no extra copies in RAM) n={len(y_train)}",
            flush=True,
        )
        x_tr = x_train
        y_tr = y_train
        train_ds = _OnlineSeq(x_train, y_train, repeats, seed)
    else:
        print(f"  building {aug_copies} aug copies...", flush=True)
        aug_x, aug_y = [x_train], [y_train]
        for k in range(aug_copies):
            aug_x.append(augment_batch(x_train, rng))
            aug_y.append(y_train)
            print(f"  aug copy {k + 1}/{aug_copies}", flush=True)
        x_tr = np.concatenate(aug_x, axis=0)
        y_tr = np.concatenate(aug_y, axis=0)
        del aug_x, aug_y
        print(f"  train after aug: {x_tr.shape}", flush=True)
        train_ds = TensorDataset(
            torch.tensor(x_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}", flush=True)
    model = _build_torch_model(
        num_classes,
        x_train.shape[-1],
        d_model=d_model,
        nhead=nhead,
        layers=layers,
    ).to(device)

    # Class weights
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = np.clip(weights, 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin, num_workers=0)

    best_acc = -1.0
    best_state = None
    stall = 0
    history = {"train_acc": [], "val_acc": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            # light online noise + mixup
            if rng.random() < 0.5:
                xb = xb + torch.randn_like(xb) * 0.015
            if rng.random() < 0.15:
                lam = float(rng.beta(0.4, 0.4))
                idx = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1.0 - lam) * xb[idx]
                logits = model(xb)
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb[idx])
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum())
            total += len(yb)
        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = v_total = 0
        v_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                logits = model(xb)
                v_loss_sum += float(criterion(logits, yb)) * len(yb)
                pred = logits.argmax(dim=1)
                v_correct += int((pred == yb).sum())
                v_total += len(yb)
        val_acc = v_correct / max(v_total, 1)
        val_loss = v_loss_sum / max(v_total, 1)
        sched.step(val_acc)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"  epoch {epoch:03d}  train={train_acc:.3f}  val={val_acc:.3f}  loss={val_loss:.3f}", flush=True)

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  early stop at epoch {epoch} (best val={best_acc:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_accuracy": best_acc, "history": history}


def train_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    aug_copies: int = 8,
    seed: int = 42,
) -> Tuple[object, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    print(f"  HGB building features (aug_copies={aug_copies})...", flush=True)
    feats, labels = [], []
    for copy_i in range(aug_copies + 1):
        batch = x_train if copy_i == 0 else augment_batch(x_train, rng)
        for i in range(len(batch)):
            feats.append(temporal_stats(batch[i]))
            labels.append(int(y_train[i]))
        print(f"  HGB aug {copy_i}/{aug_copies}", flush=True)
    X = np.stack(feats)
    y = np.asarray(labels)
    Xv = np.stack([temporal_stats(x) for x in x_val])

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=300,
                    l2_regularization=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    acc = float(clf.score(Xv, y_val))
    print(f"  HGB val accuracy={acc:.3f}")
    return clf, acc


def predict_transformer(model, x: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def train_landmark_models(
    processed_dir: str = DEFAULT_LANDMARK_DIR,
    output_dir: str = "models/recognition",
    epochs: int = 80,
    aug_copies: int = 6,
    seed: int = 42,
) -> dict:
    import joblib
    import torch

    processed = Path(processed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_names"])
    num_classes = len(class_names)

    print("Loading landmark sequences...")
    x_train, y_train = _load_split(processed, "train")
    x_val, y_val = _load_split(processed, "val")
    print(f"train={x_train.shape} val={x_val.shape} classes={num_classes}")

    print("Training Transformer...")
    model, t_report = train_transformer(
        x_train,
        y_train,
        x_val,
        y_val,
        num_classes,
        epochs=epochs,
        aug_copies=aug_copies,
        seed=seed,
    )
    t_probs = predict_transformer(model, x_val)
    t_pred = t_probs.argmax(axis=1)
    t_acc = float((t_pred == y_val).mean())

    print("Training HistGradientBoosting ensemble branch...")
    hgb, h_acc = train_hgb(x_train, y_train, x_val, y_val, aug_copies=aug_copies + 2, seed=seed)
    h_probs = hgb.predict_proba(np.stack([temporal_stats(x) for x in x_val]))

    # Soft ensemble — weight transformer higher when it wins
    tw = 0.85 if t_acc >= h_acc else 0.55
    blend = tw * t_probs + (1.0 - tw) * h_probs
    b_pred = blend.argmax(axis=1)
    b_acc = float((b_pred == y_val).mean())
    # Prefer pure transformer if ensemble does not help
    if t_acc >= b_acc:
        blend = t_probs
        b_pred = t_pred
        b_acc = t_acc
        tw = 1.0
    print(
        f"Ensemble val accuracy={b_acc:.3f} "
        f"(transformer={t_acc:.3f}, hgb={h_acc:.3f}, tw={tw:.2f})",
        flush=True,
    )

    # Per-class recall
    recalls = []
    for c in range(num_classes):
        mask = y_val == c
        if mask.any():
            recalls.append(float((b_pred[mask] == c).mean()))
        else:
            recalls.append(0.0)
    macro_recall = float(np.mean(recalls))

    pt_path = out / MODEL_PT
    sk_path = out / MODEL_SKLEARN
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_dim": int(x_train.shape[-1]),
            "num_frames": int(x_train.shape[1]),
            "num_classes": num_classes,
            "class_names": class_names,
            "d_model": 128,
            "nhead": 4,
            "layers": 3,
        },
        pt_path,
    )
    joblib.dump(hgb, sk_path)

    report = {
        "backend": "wlasl_landmarks",
        "architecture": "pose+hands Transformer + HistGradientBoosting ensemble",
        "dataset": "Voxel51/WLASL top-K (landmark sequences)",
        "num_classes": num_classes,
        "class_names": class_names,
        "num_frames": int(x_train.shape[1]),
        "feat_dim": int(x_train.shape[-1]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "val_accuracy": b_acc,
        "transformer_val_accuracy": t_acc,
        "hgb_val_accuracy": h_acc,
        "val_macro_recall": macro_recall,
        "per_class_recall": {class_names[i]: recalls[i] for i in range(num_classes)},
        "transformer_path": str(pt_path),
        "hgb_path": str(sk_path),
        "transformer_weight": tw,
        "inference": "landmark_transformer" if tw >= 0.99 else "landmark_ensemble",
        "target_signs": 50,
        "target_accuracy": 0.85,
        "signs_met": num_classes >= 50,
        "accuracy_met": b_acc >= 0.85,
        "history": t_report.get("history"),
    }
    # Keep filename expected by live/web WLASL path; mark as landmark ensemble.
    (out / METADATA).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("val_accuracy", "transformer_val_accuracy", "hgb_val_accuracy", "accuracy_met", "signs_met")}, indent=2))
    return report
