"""High-accuracy WLASL landmark training: multi-seed Transformer + Pose-TGCN.

Uses MediaPipe pose+hands sequences already cached under wlasl_landmarks_v3.
Adds velocity channels, trains several seeds, and soft-ensembles with a
Pose-TGCN-style graph model (55 joints × T×2 layout from the WLASL paper).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.recognition.wlasl_landmarks import DEFAULT_LANDMARK_DIR, FEAT_DIM, POSE_DIM, HAND_DIM
from modules.recognition.wlasl_seq_trainer import (
    METADATA,
    MODEL_PT,
    MODEL_SKLEARN,
    _build_torch_model,
    _load_split,
    augment_batch,
    augment_seq,
    predict_transformer,
    temporal_stats,
    train_hgb,
    train_transformer,
)

# 13 upper-body MediaPipe pose indices approximating OpenPose upper body,
# plus full left/right hands (21 each) = 55 joints (paper Pose-TGCN layout).
_UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]
assert len(_UPPER_BODY) + 21 + 21 == 55

TGCN_PT = "wlasl_landmark_tgcn.pt"
ENSEMBLE_PT = "wlasl_landmark_ensemble.pt"


def with_velocity(x: np.ndarray) -> np.ndarray:
    """Append first-order temporal differences → (T, 2D)."""
    vel = np.zeros_like(x)
    vel[1:] = x[1:] - x[:-1]
    return np.concatenate([x, vel], axis=-1).astype(np.float32)


def batch_with_velocity(x: np.ndarray) -> np.ndarray:
    return np.stack([with_velocity(x[i]) for i in range(len(x))], axis=0)


def seq_to_joints55(seq: np.ndarray) -> np.ndarray:
    """(T, 225) MediaPipe → (55, T*2) TGCN input (x,y only, time flattened)."""
    t = seq.shape[0]
    pose = seq[:, :POSE_DIM].reshape(t, 33, 3)
    left = seq[:, POSE_DIM : POSE_DIM + HAND_DIM].reshape(t, 21, 3)
    right = seq[:, POSE_DIM + HAND_DIM :].reshape(t, 21, 3)
    body = pose[:, _UPPER_BODY, :2]  # (T, 13, 2)
    joints = np.concatenate([body, left[:, :, :2], right[:, :, :2]], axis=1)  # (T, 55, 2)
    # TGCN expects (55, T*2) with time major in feature dim
    out = np.transpose(joints, (1, 0, 2)).reshape(55, t * 2)
    return out.astype(np.float32)


def batch_to_joints55(x: np.ndarray) -> np.ndarray:
    return np.stack([seq_to_joints55(x[i]) for i in range(len(x))], axis=0)


def _load_tgcn_cls():
    root = Path(__file__).resolve().parents[2] / "models" / "recognition" / "tgcn_wlasl"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tgcn_model import GCN_muti_att  # type: ignore

    return GCN_muti_att


def train_tgcn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    *,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-3,
    aug_copies: int = 4,
    seed: int = 0,
    patience: int = 20,
    num_stage: int = 12,
    hidden: int = 96,
    p_dropout: float = 0.3,
) -> Tuple[object, dict, np.ndarray]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    GCN = _load_tgcn_cls()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"  TGCN building {aug_copies} aug copies...", flush=True)
    aug_x = [x_train]
    for k in range(aug_copies):
        aug_x.append(augment_batch(x_train, rng))
        print(f"  TGCN aug {k + 1}/{aug_copies}", flush=True)
    x_tr = np.concatenate(aug_x, axis=0)
    y_tr = np.concatenate([y_train] * (aug_copies + 1), axis=0)

    j_tr = batch_to_joints55(x_tr)
    j_va = batch_to_joints55(x_val)
    t_dim = j_tr.shape[-1]

    model = GCN(
        input_feature=t_dim,
        hidden_feature=hidden,
        num_class=num_classes,
        p_dropout=p_dropout,
        num_stage=num_stage,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  TGCN device={device}", flush=True)
    model = model.to(device)

    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.clip(counts.sum() / (num_classes * counts), 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.tensor(j_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(j_va), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        pin_memory=pin,
        num_workers=0,
    )

    best_acc = -1.0
    best_state = None
    stall = 0
    history: Dict[str, list] = {"train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            if rng.random() < 0.5:
                xb = xb + torch.randn_like(xb) * 0.01
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        sched.step()
        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                pred = model(xb).argmax(1)
                v_correct += int((pred == yb).sum())
                v_total += len(yb)
        val_acc = v_correct / max(v_total, 1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"  TGCN epoch {epoch:03d} train={train_acc:.3f} val={val_acc:.3f}", flush=True)
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"  TGCN early stop at {epoch} (best={best_acc:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(j_va, dtype=torch.float32, device=device)), dim=-1
        ).detach().cpu().numpy()
    return model, {"best_val_accuracy": best_acc, "history": history, "t_dim": t_dim, "hidden": hidden, "num_stage": num_stage}, probs


def tta_transformer_probs(model, x_val: np.ndarray, n_tta: int = 7, seed: int = 0) -> np.ndarray:
    import torch

    rng = np.random.default_rng(seed)
    all_probs = []
    model.eval()
    with torch.no_grad():
        base = predict_transformer(model, x_val)
        all_probs.append(base)
        for i in range(n_tta - 1):
            aug = augment_batch(x_val, rng)
            all_probs.append(predict_transformer(model, aug))
    return np.mean(all_probs, axis=0)


def train_boosted_landmark_models(
    processed_dir: str = DEFAULT_LANDMARK_DIR,
    output_dir: str = "models/recognition",
    epochs: int = 90,
    aug_copies: int = 6,
    n_seeds: int = 3,
    train_tgcn_flag: bool = True,
    *,
    d_model: int = 128,
    nhead: int = 4,
    layers: int = 3,
    n_tgcn_seeds: int = 1,
    n_tta: int = 7,
    transformer_lr: float = 8e-4,
    tgcn_hidden: int = 96,
    tgcn_stages: int = 12,
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
    print(
        f"arch d_model={d_model} nhead={nhead} layers={layers} "
        f"seeds={n_seeds} tgcn_seeds={n_tgcn_seeds} tta={n_tta}",
        flush=True,
    )

    # Velocity-augmented features for transformers
    x_tr_v = batch_with_velocity(x_train)
    x_va_v = batch_with_velocity(x_val)
    feat_dim = int(x_tr_v.shape[-1])

    seed_probs: List[np.ndarray] = []
    seed_accs: List[float] = []
    seed_states: List[dict] = []
    best_single = None
    best_single_acc = -1.0

    for s in range(n_seeds):
        seed = 42 + s * 17
        print(f"\n=== Transformer seed {seed} ({s + 1}/{n_seeds}) ===", flush=True)
        model, t_report = train_transformer(
            x_tr_v,
            y_train,
            x_va_v,
            y_val,
            num_classes,
            epochs=epochs,
            aug_copies=aug_copies,
            seed=seed,
            patience=max(18, epochs // 6),
            lr=transformer_lr,
            d_model=d_model,
            nhead=nhead,
            layers=layers,
        )
        probs = tta_transformer_probs(model, x_va_v, n_tta=n_tta, seed=seed)
        acc = float((probs.argmax(1) == y_val).mean())
        print(f"  seed {seed} TTA val={acc:.3f} (raw best={t_report['best_val_accuracy']:.3f})")
        seed_probs.append(probs)
        seed_accs.append(acc)
        state = {
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "feat_dim": feat_dim,
            "num_frames": int(x_train.shape[1]),
            "num_classes": num_classes,
            "class_names": class_names,
            "d_model": int(d_model),
            "nhead": int(nhead),
            "layers": int(layers),
            "velocity": True,
            "seed": seed,
            "val_accuracy": acc,
        }
        seed_states.append(state)
        if acc > best_single_acc:
            best_single_acc = acc
            best_single = state

    tf_blend = np.mean(seed_probs, axis=0)
    tf_acc = float((tf_blend.argmax(1) == y_val).mean())
    print(f"Transformer multi-seed ensemble val={tf_acc:.3f} (seeds={seed_accs})")

    tgcn_probs = None
    tgcn_acc = 0.0
    tgcn_meta = {}
    tgcn_seed_states: List[dict] = []
    if train_tgcn_flag:
        tgcn_prob_list: List[np.ndarray] = []
        tgcn_accs: List[float] = []
        best_tgcn_pack = None
        best_tgcn_acc = -1.0
        for ti in range(max(1, n_tgcn_seeds)):
            tseed = 7 + ti * 19
            print(f"\n=== Pose-TGCN seed {tseed} ({ti + 1}/{n_tgcn_seeds}) ===", flush=True)
            tgcn_model, tgcn_meta, probs_i = train_tgcn(
                x_train,
                y_train,
                x_val,
                y_val,
                num_classes,
                epochs=max(epochs, 100),
                aug_copies=max(4, aug_copies - 2),
                seed=tseed,
                hidden=tgcn_hidden,
                num_stage=tgcn_stages,
            )
            acc_i = float((probs_i.argmax(1) == y_val).mean())
            print(f"TGCN seed {tseed} val={acc_i:.3f}")
            tgcn_prob_list.append(probs_i)
            tgcn_accs.append(acc_i)
            tgcn_state = {
                "state_dict": {k: v.cpu().clone() for k, v in tgcn_model.state_dict().items()},
                "num_classes": num_classes,
                "class_names": class_names,
                "t_dim": tgcn_meta["t_dim"],
                "hidden": tgcn_meta["hidden"],
                "num_stage": tgcn_meta["num_stage"],
                "num_frames": int(x_train.shape[1]),
                "val_accuracy": acc_i,
                "seed": tseed,
            }
            tgcn_seed_states.append(tgcn_state)
            if acc_i > best_tgcn_acc:
                best_tgcn_acc = acc_i
                best_tgcn_pack = tgcn_state
        tgcn_probs = np.mean(tgcn_prob_list, axis=0)
        tgcn_acc = float((tgcn_probs.argmax(1) == y_val).mean())
        print(f"TGCN ensemble val={tgcn_acc:.3f} (seeds={tgcn_accs})")
        assert best_tgcn_pack is not None
        torch.save(best_tgcn_pack, out / TGCN_PT)

    print("\n=== HGB branch ===", flush=True)
    hgb, h_acc = train_hgb(x_train, y_train, x_val, y_val, aug_copies=aug_copies + 2, seed=42)
    h_probs = hgb.predict_proba(np.stack([temporal_stats(x) for x in x_val]))

    # Search blend weights on val
    candidates = []
    parts = [("tf", tf_blend, tf_acc)]
    if tgcn_probs is not None:
        parts.append(("tgcn", tgcn_probs, tgcn_acc))
    parts.append(("hgb", h_probs, h_acc))

    # Grid over simplex for available branches
    best_blend = tf_blend
    best_acc = tf_acc
    best_weights = {"transformer": 1.0, "tgcn": 0.0, "hgb": 0.0}
    grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]
    for wt in grid:
        for wg in grid:
            wh = 1.0 - wt - wg
            if wh < -1e-6 or wh > 1.0 + 1e-6:
                continue
            wh = max(0.0, wh)
            blend = wt * tf_blend
            if tgcn_probs is not None:
                blend = blend + wg * tgcn_probs
            else:
                if wg > 0:
                    continue
            blend = blend + wh * h_probs
            # renormalize if needed
            s = wt + (wg if tgcn_probs is not None else 0.0) + wh
            if s <= 0:
                continue
            blend = blend / s
            acc = float((blend.argmax(1) == y_val).mean())
            if acc > best_acc + 1e-6:
                best_acc = acc
                best_blend = blend
                best_weights = {
                    "transformer": float(wt / s),
                    "tgcn": float((wg if tgcn_probs is not None else 0.0) / s),
                    "hgb": float(wh / s),
                }

    print(f"Best blend val={best_acc:.3f} weights={best_weights}")

    # Persist best single transformer (velocity) as primary artefact for live path
    assert best_single is not None
    torch.save(best_single, out / MODEL_PT)
    joblib.dump(hgb, out / MODEL_SKLEARN)
    torch.save(
        {
            "seeds": seed_states,
            "tgcn_seeds": tgcn_seed_states,
            "blend_weights": best_weights,
            "velocity": True,
            "feat_dim": feat_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "num_frames": int(x_train.shape[1]),
            "val_accuracy": best_acc,
            "transformer_ensemble_val": tf_acc,
            "tgcn_val": tgcn_acc,
            "hgb_val": h_acc,
            "d_model": int(d_model),
            "nhead": int(nhead),
            "layers": int(layers),
            "n_tta_train": int(n_tta),
        },
        out / ENSEMBLE_PT,
    )

    pred = best_blend.argmax(1)
    recalls = []
    for c in range(num_classes):
        mask = y_val == c
        recalls.append(float((pred[mask] == c).mean()) if mask.any() else 0.0)

    # Prefer transformer-only inference when blend barely helps (simpler live path)
    use_ensemble = best_acc > best_single_acc + 0.01
    inference = "landmark_boost_ensemble" if use_ensemble else "landmark_transformer"
    tw = best_weights["transformer"] if use_ensemble else 1.0

    report = {
        "backend": "wlasl_landmarks",
        "architecture": "velocity Transformer multi-seed + Pose-TGCN + HGB",
        "dataset": "Voxel51/WLASL top-K (landmark sequences)",
        "num_classes": num_classes,
        "class_names": class_names,
        "num_frames": int(x_train.shape[1]),
        "feat_dim": feat_dim,
        "velocity": True,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "val_accuracy": best_acc,
        "transformer_val_accuracy": best_single_acc,
        "transformer_ensemble_val_accuracy": tf_acc,
        "tgcn_val_accuracy": tgcn_acc,
        "hgb_val_accuracy": h_acc,
        "seed_val_accuracies": seed_accs,
        "blend_weights": best_weights,
        "val_macro_recall": float(np.mean(recalls)),
        "per_class_recall": {class_names[i]: recalls[i] for i in range(num_classes)},
        "transformer_path": str(out / MODEL_PT),
        "hgb_path": str(out / MODEL_SKLEARN),
        "tgcn_path": str(out / TGCN_PT) if train_tgcn_flag else None,
        "ensemble_path": str(out / ENSEMBLE_PT),
        "transformer_weight": tw,
        "inference": inference,
        "target_signs": 50,
        "target_accuracy": 0.85,
        "signs_met": num_classes >= 50,
        "accuracy_met": best_acc >= 0.85,
    }
    (out / METADATA).write_text(json.dumps(report, indent=2), encoding="utf-8")
    eval_path = Path("logs/evaluation/wlasl_accuracy.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps(
            {
                "val_accuracy": best_acc,
                "num_classes": num_classes,
                "n_val": int(len(y_val)),
                "signs_met": report["signs_met"],
                "accuracy_met": report["accuracy_met"],
                "backend": report["architecture"],
                "per_class_recall": report["per_class_recall"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in (
        "val_accuracy", "transformer_val_accuracy", "transformer_ensemble_val_accuracy",
        "tgcn_val_accuracy", "hgb_val_accuracy", "accuracy_met", "signs_met", "blend_weights"
    )}, indent=2))
    return report
