"""Chase v4: richer features + extra branches + frozen v3 expert.

Goal is to beat chase v3 (69.2%) and push toward the 85% objective.
Methods that are new vs v3:
- hand-bone geometry channels
- horizontal mirror + time-warp, extra copies of historically weak glosses
- CNN-BiLSTM branch (different inductive bias from Transformer / TGCN)
- distance-weighted kNN on pooled features
- frozen chase-v3 ensemble as an extra voter (same 50-gloss split)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.recognition.wlasl_boost_trainer import (
    _load_tgcn_cls,
    batch_to_joints55,
    batch_with_velocity,
    train_tgcn,
    tta_transformer_probs,
    with_velocity,
)
from modules.recognition.wlasl_landmarks import HAND_DIM, POSE_DIM
from modules.recognition.wlasl_seq_trainer import (
    METADATA,
    MODEL_PT,
    MODEL_SKLEARN,
    _build_torch_model,
    _load_split,
    temporal_stats,
    train_hgb,
    train_transformer,
)

# MediaPipe pose left/right pairs (swap on horizontal mirror).
_POSE_SWAP = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)
# 20 hand bones (MediaPipe hand topology).
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
# Glosses with the weakest recall on chase v3. Oversampled, not dropped.
_WEAK_GLOSSES = {
    "before", "go", "computer", "cousin", "bed", "thanksgiving",
    "call", "cold", "corn", "last", "later", "shirt",
}

CNN_PT = "wlasl_landmark_cnn.pt"
KNN_PT = "wlasl_landmark_knn.joblib"
V3_EXPERT_SUBDIR = "experts/v3"


def build_cnn_lstm(num_classes: int, feat_dim: int):
    import torch
    from torch import nn

    class CnnLstm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(feat_dim, 160, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Conv1d(160, 160, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.lstm = nn.LSTM(
                160, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, num_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.lstm(h)
            return self.head(h.mean(dim=1))

    return CnnLstm


class FrozenV3ExpertRuntime:
    """Chase-v3 expert loaded once for live inference (avoids reload every word)."""

    def __init__(
        self,
        *,
        seed_models: List[Tuple[object, int]],
        tgcn_models: List[object],
        hgb: Optional[object],
        blend_weights: dict,
    ) -> None:
        self._seed_models = seed_models
        self._tgcn_models = tgcn_models
        self._hgb = hgb
        self._blend_weights = blend_weights

    @classmethod
    def load(
        cls, expert_dir: Path, class_names: Sequence[str]
    ) -> Optional["FrozenV3ExpertRuntime"]:
        import torch

        ens_path = expert_dir / "wlasl_landmark_ensemble.pt"
        if not ens_path.exists():
            return None
        ens = torch.load(ens_path, map_location="cpu", weights_only=False)
        if list(ens.get("class_names") or []) != list(class_names):
            return None

        seed_models: List[Tuple[object, int]] = []
        for state in ens.get("seeds") or []:
            model = _build_torch_model(
                num_classes=int(state["num_classes"]),
                feat_dim=int(state.get("feat_dim", 450)),
                d_model=int(state.get("d_model", 128)),
                nhead=int(state.get("nhead", 4)),
                layers=int(state.get("layers", 3)),
            )
            model.load_state_dict(state["state_dict"])
            model.eval()
            seed_models.append((model, int(state.get("seed", 0))))
        if not seed_models:
            return None

        tgcn_models: List[object] = []
        tgcn_states = list(ens.get("tgcn_seeds") or [])
        if tgcn_states:
            GCN = _load_tgcn_cls()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for state in tgcn_states:
                gcn = GCN(
                    input_feature=int(state["t_dim"]),
                    hidden_feature=int(state.get("hidden", 96)),
                    num_class=int(state["num_classes"]),
                    p_dropout=0.3,
                    num_stage=int(state.get("num_stage", 12)),
                )
                gcn.load_state_dict(state["state_dict"])
                gcn.eval()
                gcn.to(device)
                tgcn_models.append(gcn)

        hgb = None
        hgb_path = expert_dir / "wlasl_landmark_hgb.joblib"
        if hgb_path.exists():
            import joblib

            try:
                hgb = joblib.load(hgb_path)
            except Exception as exc:
                print(f"  frozen HGB skipped: {exc}", flush=True)

        return cls(
            seed_models=seed_models,
            tgcn_models=tgcn_models,
            hgb=hgb,
            blend_weights=dict(ens.get("blend_weights") or {"transformer": 1.0}),
        )

    def predict_batch(self, x_val: np.ndarray) -> Optional[np.ndarray]:
        import torch

        x_val = np.asarray(x_val, dtype=np.float32)
        x_v = batch_with_velocity(x_val)
        seed_probs = []
        for model, seed in self._seed_models:
            seed_probs.append(tta_transformer_probs(model, x_v, n_tta=5, seed=seed))
        tf = np.mean(seed_probs, axis=0)

        tgcn_probs = None
        if self._tgcn_models:
            device = next(self._tgcn_models[0].parameters()).device
            j = torch.tensor(batch_to_joints55(x_val), dtype=torch.float32, device=device)
            parts = []
            with torch.no_grad():
                for gcn in self._tgcn_models:
                    parts.append(torch.softmax(gcn(j), dim=-1).cpu().numpy())
            tgcn_probs = np.mean(parts, axis=0)

        hgb_probs = None
        if self._hgb is not None:
            feats = np.stack([temporal_stats(x_val[i]) for i in range(len(x_val))])
            raw = self._hgb.predict_proba(feats)
            classes = (
                self._hgb.named_steps["hgb"].classes_
                if hasattr(self._hgb, "named_steps")
                else self._hgb.classes_
            )
            hgb_probs = np.zeros_like(tf)
            for i, c in enumerate(classes):
                hgb_probs[:, int(c)] = raw[:, i]

        bw = self._blend_weights
        wt = float(bw.get("transformer", 0.0))
        wg = float(bw.get("tgcn", 0.0)) if tgcn_probs is not None else 0.0
        wh = float(bw.get("hgb", 0.0)) if hgb_probs is not None else 0.0
        total = wt + wg + wh
        if total <= 0:
            return tf
        blend = wt * tf
        if tgcn_probs is not None:
            blend = blend + wg * tgcn_probs
        if hgb_probs is not None:
            blend = blend + wh * hgb_probs
        return (blend / total).astype(np.float32)


def frozen_v3_probs_single(
    expert_dir: Path,
    seq: np.ndarray,
    class_names: Sequence[str],
    *,
    v3_runtime: Optional[FrozenV3ExpertRuntime] = None,
) -> Optional[np.ndarray]:
    batch_in = np.expand_dims(seq.astype(np.float32), 0)
    if v3_runtime is not None:
        batch = v3_runtime.predict_batch(batch_in)
    else:
        batch = frozen_v3_probs(expert_dir, batch_in, class_names)
    if batch is None:
        return None
    return batch[0]


def predict_chase_v4_probs(
    seq: np.ndarray,
    *,
    class_names: Sequence[str],
    blend_weights: dict,
    torch_models: Sequence[object],
    tgcn_models: Sequence[object],
    cnn_model: Optional[object],
    hgb: Optional[object],
    knn: Optional[object],
    v3_expert_dir: Optional[Path],
    v3_runtime: Optional[FrozenV3ExpertRuntime] = None,
) -> np.ndarray:
    """Single-sequence inference matching Colab v4 blend search."""
    import torch

    from modules.recognition.wlasl_boost_trainer import batch_to_joints55
    from modules.recognition.wlasl_seq_trainer import augment_seq, temporal_stats

    raw = seq[:, :225].astype(np.float32) if seq.shape[-1] >= 225 else seq.astype(np.float32)
    rich = rich_seq(raw)
    n_cls = len(class_names)
    parts: Dict[str, np.ndarray] = {}

    xs = [rich]
    rng = np.random.default_rng(0)
    for _ in range(4):
        xs.append(augment_seq(rich, rng))
    batch = np.stack(xs, axis=0)
    tta = []
    with torch.no_grad():
        for model in torch_models:
            logits = model(torch.tensor(batch, dtype=torch.float32))
            tta.append(torch.softmax(logits, dim=-1).numpy().mean(axis=0))
    parts["transformer"] = np.mean(tta, axis=0) if tta else np.zeros(n_cls, dtype=np.float32)

    if tgcn_models:
        j = batch_to_joints55(np.expand_dims(raw, 0))
        g_list = []
        with torch.no_grad():
            for m in tgcn_models:
                g_list.append(
                    torch.softmax(m(torch.tensor(j, dtype=torch.float32)), dim=-1).numpy()[0]
                )
        parts["tgcn"] = np.mean(g_list, axis=0)

    if cnn_model is not None:
        with torch.no_grad():
            logits = cnn_model(torch.tensor(rich, dtype=torch.float32).unsqueeze(0))
            parts["cnn"] = torch.softmax(logits, dim=-1).numpy()[0]

    if hgb is not None:
        raw_h = hgb.predict_proba(np.expand_dims(temporal_stats(rich), axis=0))[0]
        h_probs = np.zeros(n_cls, dtype=np.float32)
        classes = hgb.named_steps["hgb"].classes_ if hasattr(hgb, "named_steps") else hgb.classes_
        for i, c in enumerate(classes):
            h_probs[int(c)] = raw_h[i]
        parts["hgb"] = h_probs

    if knn is not None:
        raw_k = knn.predict_proba(np.expand_dims(temporal_stats(rich), axis=0))[0]
        k_probs = np.zeros(n_cls, dtype=np.float32)
        classes = knn.named_steps["knn"].classes_
        for i, c in enumerate(classes):
            k_probs[int(c)] = raw_k[i]
        parts["knn"] = k_probs

    if float(blend_weights.get("v3", 0.0)) > 0 and (
        v3_runtime is not None or v3_expert_dir is not None
    ):
        v3 = frozen_v3_probs_single(
            v3_expert_dir or Path("."),
            raw,
            class_names,
            v3_runtime=v3_runtime,
        )
        if v3 is not None:
            parts["v3"] = v3

    blend = np.zeros(n_cls, dtype=np.float64)
    total = 0.0
    for key, w in blend_weights.items():
        w = float(w)
        if w <= 0 or key not in parts:
            continue
        blend += w * parts[key]
        total += w
    if total <= 0:
        return parts.get("transformer", np.ones(n_cls) / n_cls)
    return (blend / total).astype(np.float32)


def mirror_seq(seq: np.ndarray) -> np.ndarray:
    """Flip hip-centred landmarks across the body midline and swap L/R."""
    seq = seq[:, : POSE_DIM + 2 * HAND_DIM].astype(np.float32, copy=True)
    t = seq.shape[0]
    pose = seq[:, :POSE_DIM].reshape(t, 33, 3)
    left = seq[:, POSE_DIM : POSE_DIM + HAND_DIM].reshape(t, 21, 3)
    right = seq[:, POSE_DIM + HAND_DIM :].reshape(t, 21, 3)
    pose[..., 0] *= -1
    left[..., 0] *= -1
    right[..., 0] *= -1
    for a, b in _POSE_SWAP:
        pose[:, [a, b]] = pose[:, [b, a]]
    left, right = right.copy(), left.copy()
    return np.concatenate(
        [pose.reshape(t, -1), left.reshape(t, -1), right.reshape(t, -1)], axis=-1
    ).astype(np.float32)


def time_warp(seq: np.ndarray, rate: float) -> np.ndarray:
    """Resample time. rate>1 compresses the motion toward the middle."""
    t, d = seq.shape
    src = np.arange(t, dtype=np.float32)
    centre = (t - 1) / 2.0
    dst = np.clip((src - centre) * rate + centre, 0, t - 1)
    out = np.stack([np.interp(dst, src, seq[:, i]) for i in range(d)], axis=1)
    return out.astype(np.float32)


def _hand_bones(hand: np.ndarray) -> np.ndarray:
    cols = [np.linalg.norm(hand[:, a] - hand[:, b], axis=-1) for a, b in _HAND_EDGES]
    return np.stack(cols, axis=-1).astype(np.float32)


def geometry_seq(seq: np.ndarray) -> np.ndarray:
    """Per-frame bone lengths + inter-hand distance. (T, 225) → (T, 41)."""
    t = seq.shape[0]
    pose = seq[:, :POSE_DIM].reshape(t, 33, 3)
    left = seq[:, POSE_DIM : POSE_DIM + HAND_DIM].reshape(t, 21, 3)
    right = seq[:, POSE_DIM + HAND_DIM : POSE_DIM + 2 * HAND_DIM].reshape(t, 21, 3)
    bones = np.concatenate([_hand_bones(left), _hand_bones(right)], axis=-1)
    gap = np.linalg.norm(left[:, 0] - right[:, 0], axis=-1, keepdims=True).astype(np.float32)
    return np.concatenate([bones, gap], axis=-1)


def rich_seq(seq: np.ndarray) -> np.ndarray:
    """Landmarks + geometry + velocity of that concat."""
    base = seq[:, : POSE_DIM + 2 * HAND_DIM].astype(np.float32)
    return with_velocity(np.concatenate([base, geometry_seq(base)], axis=-1))


def rich_batch(x: np.ndarray) -> np.ndarray:
    return np.stack([rich_seq(x[i]) for i in range(len(x))], axis=0)


def expand_train(
    x: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = [x]
    ys: List[np.ndarray] = [y]
    mirrored = np.stack([mirror_seq(x[i]) for i in range(len(x))], axis=0)
    xs.append(mirrored)
    ys.append(y)
    weak_idx = [i for i, yi in enumerate(y) if class_names[int(yi)] in _WEAK_GLOSSES]
    if weak_idx:
        base = x[weak_idx]
        xs.append(np.stack([time_warp(s, 0.75) for s in base], axis=0))
        ys.append(y[weak_idx])
        xs.append(np.stack([time_warp(s, 1.3) for s in base], axis=0))
        ys.append(y[weak_idx])
        xs.append(np.stack([mirror_seq(time_warp(s, 0.85)) for s in base], axis=0))
        ys.append(y[weak_idx])
    out_x = np.concatenate(xs, axis=0)
    out_y = np.concatenate(ys, axis=0)
    print(f"  expanded train {x.shape} → {out_x.shape} (mirror + weak-class warps)", flush=True)
    return out_x, out_y


def train_cnn_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    *,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 3,
    patience: int = 16,
    aug_copies: int = 3,
) -> Tuple[object, np.ndarray, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from modules.recognition.wlasl_seq_trainer import augment_batch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    feat_dim = int(x_train.shape[-1])

    copies = [x_train]
    labels = [y_train]
    for _ in range(aug_copies):
        copies.append(augment_batch(x_train, rng))
        labels.append(y_train)
    x_tr = np.concatenate(copies, axis=0)
    y_tr = np.concatenate(labels, axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  CNN-LSTM device={device} train={x_tr.shape}", flush=True)
    model = build_cnn_lstm(num_classes, feat_dim)().to(device)
    counts = np.maximum(np.bincount(y_tr, minlength=num_classes).astype(np.float64), 1.0)
    weights = np.clip(counts.sum() / (num_classes * counts), 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pin = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32)
    best_acc = -1.0
    best_state = None
    stall = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_x.to(device)).argmax(1).cpu().numpy()
        acc = float((pred == y_val).mean())
        print(f"  CNN-LSTM epoch {epoch:03d} val={acc:.3f}", flush=True)
        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  CNN-LSTM early stop at {epoch} (best={best_acc:.3f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(val_x.to(device)), dim=-1).cpu().numpy()
    return model, probs, best_acc


def frozen_v3_probs(
    expert_dir: Path,
    x_val: np.ndarray,
    class_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Replay chase-v3 blend on raw 225-d val sequences."""
    ens_path = expert_dir / "wlasl_landmark_ensemble.pt"
    if not ens_path.exists():
        print(f"  no frozen expert at {ens_path}", flush=True)
        return None
    runtime = FrozenV3ExpertRuntime.load(expert_dir, class_names)
    if runtime is None:
        print("  frozen expert class order mismatch — skipping", flush=True)
        return None
    return runtime.predict_batch(x_val)


def _search_blend(parts: Dict[str, np.ndarray], y_val: np.ndarray, n_random: int = 500) -> Tuple[dict, float, np.ndarray]:
    names = list(parts)
    mats = [parts[n] for n in names]
    rng = np.random.default_rng(0)
    best_acc = -1.0
    best_w = np.zeros(len(names), dtype=np.float64)
    best_w[0] = 1.0
    best_blend = mats[0]

    def _score(w: np.ndarray) -> Tuple[float, np.ndarray]:
        blend = sum(float(w[i]) * mats[i] for i in range(len(names)))
        return float((blend.argmax(1) == y_val).mean()), blend

    candidates = [best_w.copy()]
    for i in range(len(names)):
        w = np.zeros(len(names))
        w[i] = 1.0
        candidates.append(w)
    for _ in range(n_random):
        candidates.append(rng.dirichlet(np.ones(len(names))))
    for w in candidates:
        acc, blend = _score(w)
        if acc > best_acc + 1e-6:
            best_acc = acc
            best_w = w
            best_blend = blend
    weights = {names[i]: round(float(best_w[i]), 4) for i in range(len(names))}
    return weights, best_acc, best_blend


def train_chase_v4(
    processed_dir: str,
    output_dir: str,
    expert_dir: Optional[str] = None,
    *,
    epochs: int = 100,
    n_transformer_seeds: int = 3,
    n_tgcn_seeds: int = 2,
    aug_copies: int = 4,
) -> dict:
    import joblib
    import torch

    processed = Path(processed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_names"])
    num_classes = len(class_names)

    print("Loading landmark sequences...", flush=True)
    x_train, y_train = _load_split(processed, "train")
    x_val, y_val = _load_split(processed, "val")
    print(f"train={x_train.shape} val={x_val.shape} classes={num_classes}", flush=True)

    x_exp, y_exp = expand_train(x_train, y_train, class_names)
    print("Building rich features (landmarks + hand bones + velocity)...", flush=True)
    x_tr_r = rich_batch(x_exp)
    x_va_r = rich_batch(x_val)
    feat_dim = int(x_tr_r.shape[-1])
    print(f"rich feat_dim={feat_dim}", flush=True)

    seed_probs: List[np.ndarray] = []
    seed_accs: List[float] = []
    seed_states: List[dict] = []
    for s in range(n_transformer_seeds):
        seed = 42 + s * 17
        print(f"\n=== Rich Transformer seed {seed} ({s + 1}/{n_transformer_seeds}) ===", flush=True)
        model, t_report = train_transformer(
            x_tr_r,
            y_exp,
            x_va_r,
            y_val,
            num_classes,
            epochs=epochs,
            aug_copies=1,
            online_aug=True,
            seed=seed,
            patience=max(16, epochs // 6),
            lr=8e-4,
            d_model=128,
            nhead=4,
            layers=3,
        )
        probs = tta_transformer_probs(model, x_va_r, n_tta=7, seed=seed)
        acc = float((probs.argmax(1) == y_val).mean())
        print(f"  seed {seed} TTA val={acc:.3f} (raw best={t_report['best_val_accuracy']:.3f})")
        seed_probs.append(probs)
        seed_accs.append(acc)
        seed_states.append({
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "feat_dim": feat_dim,
            "num_frames": int(x_train.shape[1]),
            "num_classes": num_classes,
            "class_names": class_names,
            "d_model": 128,
            "nhead": 4,
            "layers": 3,
            "velocity": True,
            "geometry": True,
            "seed": seed,
            "val_accuracy": acc,
        })
    tf_blend = np.mean(seed_probs, axis=0)
    tf_acc = float((tf_blend.argmax(1) == y_val).mean())
    print(f"Rich Transformer ensemble val={tf_acc:.3f}", flush=True)

    tgcn_list: List[np.ndarray] = []
    tgcn_states: List[dict] = []
    # One capacity-matched TGCN, one more regularized TGCN.
    tgcn_cfgs = [
        {"hidden": 128, "num_stage": 14, "p_dropout": 0.35, "aug_copies": 0},
        {"hidden": 96, "num_stage": 8, "p_dropout": 0.5, "aug_copies": 0},
    ]
    for ti in range(n_tgcn_seeds):
        cfg = tgcn_cfgs[ti % len(tgcn_cfgs)]
        tseed = 11 + ti * 19
        print(f"\n=== TGCN seed {tseed} cfg={cfg} ===", flush=True)
        model, tmeta, probs = train_tgcn(
            x_exp,
            y_exp,
            x_val,
            y_val,
            num_classes,
            epochs=max(epochs, 80),
            aug_copies=cfg["aug_copies"],
            seed=tseed,
            hidden=cfg["hidden"],
            num_stage=cfg["num_stage"],
            p_dropout=cfg["p_dropout"],
            patience=18,
        )
        acc = float((probs.argmax(1) == y_val).mean())
        print(f"TGCN seed {tseed} val={acc:.3f}", flush=True)
        tgcn_list.append(probs)
        tgcn_states.append({
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "num_classes": num_classes,
            "class_names": class_names,
            "t_dim": tmeta["t_dim"],
            "hidden": tmeta["hidden"],
            "num_stage": tmeta["num_stage"],
            "num_frames": int(x_train.shape[1]),
            "val_accuracy": acc,
            "seed": tseed,
            "p_dropout": cfg["p_dropout"],
        })
    tgcn_blend = np.mean(tgcn_list, axis=0)
    tgcn_acc = float((tgcn_blend.argmax(1) == y_val).mean())
    print(f"TGCN ensemble val={tgcn_acc:.3f}", flush=True)
    torch.save(tgcn_states[int(np.argmax([s["val_accuracy"] for s in tgcn_states]))], out / "wlasl_landmark_tgcn.pt")

    print("\n=== CNN-LSTM ===", flush=True)
    cnn, cnn_probs, cnn_acc = train_cnn_lstm(
        x_tr_r, y_exp, x_va_r, y_val, num_classes, epochs=min(epochs, 80), aug_copies=0
    )
    print(f"CNN-LSTM val={cnn_acc:.3f}", flush=True)
    torch.save({
        "state_dict": {k: v.cpu().clone() for k, v in cnn.state_dict().items()},
        "feat_dim": feat_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "val_accuracy": cnn_acc,
    }, out / CNN_PT)

    print("\n=== HGB (rich temporal stats) ===", flush=True)
    hgb, h_acc = train_hgb(x_tr_r, y_exp, x_va_r, y_val, aug_copies=2, seed=42)
    raw_h = hgb.predict_proba(np.stack([temporal_stats(x) for x in x_va_r]))
    h_probs = np.zeros((len(y_val), num_classes), dtype=np.float32)
    h_classes = hgb.named_steps["hgb"].classes_
    for i, c in enumerate(h_classes):
        h_probs[:, int(c)] = raw_h[:, i]
    print(f"HGB val={h_acc:.3f}", flush=True)

    print("\n=== kNN ===", flush=True)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    knn_x = np.stack([temporal_stats(x_tr_r[i]) for i in range(0, len(x_tr_r), 2)])
    knn_y = y_exp[::2]
    knn_v = np.stack([temporal_stats(x) for x in x_va_r])
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")),
    ])
    knn.fit(knn_x, knn_y)
    knn_raw = knn.predict_proba(knn_v)
    knn_probs = np.zeros_like(tf_blend)
    for i, c in enumerate(knn.named_steps["knn"].classes_):
        knn_probs[:, int(c)] = knn_raw[:, i]
    knn_acc = float((knn_probs.argmax(1) == y_val).mean())
    print(f"kNN val={knn_acc:.3f}", flush=True)

    parts: Dict[str, np.ndarray] = {
        "transformer": tf_blend,
        "tgcn": tgcn_blend,
        "cnn": cnn_probs,
        "hgb": h_probs,
        "knn": knn_probs,
    }
    if expert_dir:
        print("\n=== Frozen chase-v3 expert ===", flush=True)
        v3 = frozen_v3_probs(Path(expert_dir), x_val, class_names)
        if v3 is not None:
            v3_acc = float((v3.argmax(1) == y_val).mean())
            print(f"frozen v3 val={v3_acc:.3f}", flush=True)
            parts["v3"] = v3

    print("\n=== Blend search ===", flush=True)
    weights, best_acc, best_blend = _search_blend(parts, y_val)
    print(f"Best blend val={best_acc:.3f} weights={weights}", flush=True)

    torch.save(seed_states[int(np.argmax(seed_accs))], out / MODEL_PT)
    joblib.dump(hgb, out / MODEL_SKLEARN)
    joblib.dump(knn, out / KNN_PT)
    torch.save({
        "seeds": seed_states,
        "tgcn_seeds": tgcn_states,
        "blend_weights": weights,
        "velocity": True,
        "geometry": True,
        "feat_dim": feat_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_frames": int(x_train.shape[1]),
        "val_accuracy": best_acc,
        "branch_val": {k: float((v.argmax(1) == y_val).mean()) for k, v in parts.items()},
        "has_cnn": True,
        "has_knn": True,
    }, out / "wlasl_landmark_ensemble.pt")

    pred = best_blend.argmax(1)
    recalls = []
    for c in range(num_classes):
        mask = y_val == c
        recalls.append(float((pred[mask] == c).mean()) if mask.any() else 0.0)

    report = {
        "backend": "wlasl_landmarks",
        "architecture": "chase_v4 rich Transformer + TGCN + CNN-LSTM + HGB + kNN + frozen v3",
        "dataset": "Voxel51/WLASL landmarks_v4",
        "num_classes": num_classes,
        "class_names": class_names,
        "num_frames": int(x_train.shape[1]),
        "feat_dim": feat_dim,
        "geometry": True,
        "velocity": True,
        "n_train": int(len(y_train)),
        "n_train_expanded": int(len(y_exp)),
        "n_val": int(len(y_val)),
        "val_accuracy": best_acc,
        "transformer_ensemble_val_accuracy": tf_acc,
        "tgcn_val_accuracy": tgcn_acc,
        "cnn_val_accuracy": cnn_acc,
        "hgb_val_accuracy": h_acc,
        "knn_val_accuracy": knn_acc,
        "seed_val_accuracies": seed_accs,
        "blend_weights": weights,
        "val_macro_recall": float(np.mean(recalls)),
        "per_class_recall": {class_names[i]: recalls[i] for i in range(num_classes)},
        "inference": "landmark_boost_ensemble",
        "target_accuracy": 0.85,
        "accuracy_met": best_acc >= 0.85,
        "baseline_v3": 0.6923076923076923,
        "beat_v3": best_acc > 0.6923076923076923 + 1e-6,
    }
    (out / METADATA).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "val_accuracy": best_acc,
        "accuracy_met_85": report["accuracy_met"],
        "beat_v3": report["beat_v3"],
        "blend_weights": weights,
        "branch_val": {k: float((v.argmax(1) == y_val).mean()) for k, v in parts.items()},
    }, indent=2))
    return report
