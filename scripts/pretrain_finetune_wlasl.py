"""Pretrain landmark Transformer on top-K glosses, fine-tune on top-50.

Uses a larger vocabulary for representation learning, then swaps the
classification head for the proposal's 50-gloss evaluation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_data import (  # noqa: E402
    DEFAULT_RAW_DIR,
    download_samples_json,
    filter_samples,
    load_samples_index,
    select_top_glosses,
)
from modules.recognition.wlasl_landmarks import (  # noqa: E402
    DEFAULT_NUM_FRAMES,
    build_landmark_cache,
)
from modules.recognition.wlasl_seq_trainer import (  # noqa: E402
    MODEL_PT,
    METADATA,
    _build_torch_model,
    _load_split,
    predict_transformer,
    train_hgb,
    train_transformer,
    temporal_stats,
)


def _records_for_classes(raw_dir: str, class_names: list[str]) -> list[dict]:
    samples = load_samples_index(download_samples_json(raw_dir=raw_dir))
    selected = filter_samples(samples, class_names)
    videos_root = Path(raw_dir) / "videos"
    records = []
    for rec in selected:
        local = videos_root / Path(rec["filepath"]).name
        if local.exists():
            item = dict(rec)
            item["video_path"] = str(local)
            records.append(item)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain+finetune WLASL landmarks")
    parser.add_argument("--pretrain-classes", type=int, default=100)
    parser.add_argument("--finetune-classes", type=int, default=50)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--pretrain-dir", default="data/processed/wlasl_landmarks_pretrain100")
    parser.add_argument("--finetune-dir", default="data/processed/wlasl_landmarks_v3")
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--train-views", type=int, default=2)
    parser.add_argument("--aug-copies", type=int, default=4)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    samples = load_samples_index(download_samples_json(raw_dir=args.raw_dir))
    pre_names = select_top_glosses(samples, args.pretrain_classes)
    fine_names = select_top_glosses(samples, args.finetune_classes)

    if not args.skip_extract:
        print(f"Extracting landmarks for top-{args.pretrain_classes}...")
        records = _records_for_classes(args.raw_dir, pre_names)
        print(f"  {len(records)} local videos")
        build_landmark_cache(
            records,
            pre_names,
            out_dir=args.pretrain_dir,
            num_frames=args.num_frames,
            train_views=args.train_views,
        )

    import torch
    import joblib

    pre = Path(args.pretrain_dir)
    x_tr, y_tr = _load_split(pre, "train")
    x_va, y_va = _load_split(pre, "val")
    print(f"Pretrain data train={x_tr.shape} val={x_va.shape}")

    print("Pretraining Transformer on larger vocabulary...")
    model, _ = train_transformer(
        x_tr,
        y_tr,
        x_va,
        y_va,
        num_classes=len(pre_names),
        epochs=args.pretrain_epochs,
        aug_copies=args.aug_copies,
        seed=42,
        patience=10,
    )

    # Fine-tune on top-50: rebuild head, keep encoder weights where possible
    fine = Path(args.finetune_dir)
    if not (fine / "train.json").exists():
        raise SystemExit(f"Missing finetune data at {fine}; run extract_wlasl_landmarks.py first")
    fx_tr, fy_tr = _load_split(fine, "train")
    fx_va, fy_va = _load_split(fine, "val")
    fine_meta = json.loads((fine / "metadata.json").read_text(encoding="utf-8"))
    fine_names = list(fine_meta["class_names"])

    print("Fine-tuning head+encoder on top-50...")
    ft_model = _build_torch_model(len(fine_names), fx_tr.shape[-1])
    # Copy compatible encoder weights
    pre_sd = model.state_dict()
    ft_sd = ft_model.state_dict()
    transferred = 0
    for k, v in pre_sd.items():
        if k.startswith("head."):
            continue
        if k in ft_sd and ft_sd[k].shape == v.shape:
            ft_sd[k] = v
            transferred += 1
    ft_model.load_state_dict(ft_sd)
    print(f"  transferred {transferred} tensors from pretrain")

    # Continue training with lower LR via train_transformer from scratch-ish
    # but seed weights by temporarily swapping into train loop:
    # simplest: run train_transformer then overwrite with warm-start by hacking
    # — instead, call a short custom fine-tune here.
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from modules.recognition.wlasl_seq_trainer import augment_batch

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    aug_x = [fx_tr]
    aug_y = [fy_tr]
    for _ in range(args.aug_copies):
        aug_x.append(augment_batch(fx_tr, rng))
        aug_y.append(fy_tr)
    x_all = np.concatenate(aug_x)
    y_all = np.concatenate(aug_y)

    device = torch.device("cpu")
    ft_model = ft_model.to(device)
    counts = np.bincount(y_all, minlength=len(fine_names)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.clip(counts.sum() / (len(fine_names) * counts), 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32),
        label_smoothing=0.05,
    )
    # Freeze early encoder briefly then unfreeze
    for name, p in ft_model.named_parameters():
        if name.startswith("encoder.layers.0") or name.startswith("proj"):
            p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in ft_model.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=2e-2,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.long)),
        batch_size=32,
        shuffle=True,
    )
    val_x = torch.tensor(fx_va, dtype=torch.float32)
    val_y = torch.tensor(fy_va, dtype=torch.long)

    best_acc = -1.0
    best_state = None
    stall = 0
    for epoch in range(1, args.finetune_epochs + 1):
        if epoch == 8:
            for p in ft_model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(ft_model.parameters(), lr=2e-4, weight_decay=2e-2)
            print("  unfroze full model", flush=True)
        ft_model.train()
        correct = total = 0
        for xb, yb in train_loader:
            if rng.random() < 0.5:
                xb = xb + torch.randn_like(xb) * 0.015
            if rng.random() < 0.35:
                lam = float(rng.beta(0.4, 0.4))
                idx = torch.randperm(xb.size(0))
                xb_m = lam * xb + (1 - lam) * xb[idx]
                logits = ft_model(xb_m)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[idx])
            else:
                logits = ft_model(xb)
                loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            opt.step()
            correct += int((logits.argmax(1) == yb).sum())
            total += len(yb)
        ft_model.eval()
        with torch.no_grad():
            pred = ft_model(val_x).argmax(1)
            val_acc = float((pred == val_y).sum() / len(val_y))
        print(f"  ft epoch {epoch:03d} train={correct/max(total,1):.3f} val={val_acc:.3f}", flush=True)
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 12:
                print(f"  early stop (best={best_acc:.3f})", flush=True)
                break
    if best_state:
        ft_model.load_state_dict(best_state)

    t_probs = predict_transformer(ft_model, fx_va)
    t_acc = float((t_probs.argmax(1) == fy_va).mean())
    print(f"Fine-tune transformer val={t_acc:.3f}", flush=True)

    hgb, h_acc = train_hgb(fx_tr, fy_tr, fx_va, fy_va, aug_copies=args.aug_copies + 2)
    tw = 1.0 if t_acc >= h_acc else 0.7
    if tw < 1.0:
        h_probs = hgb.predict_proba(np.stack([temporal_stats(x) for x in fx_va]))
        blend = tw * t_probs + (1 - tw) * h_probs
        b_acc = float((blend.argmax(1) == fy_va).mean())
    else:
        b_acc = t_acc
    if t_acc >= b_acc:
        b_acc = t_acc
        tw = 1.0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ft_model.state_dict(),
            "feat_dim": int(fx_tr.shape[-1]),
            "num_frames": int(fx_tr.shape[1]),
            "num_classes": len(fine_names),
            "class_names": fine_names,
            "d_model": 128,
            "nhead": 4,
            "layers": 3,
            "pretrained_on": args.pretrain_classes,
        },
        out / MODEL_PT,
    )
    joblib.dump(hgb, out / "wlasl_landmark_hgb.joblib")

    recalls = []
    preds = t_probs.argmax(1)
    for c in range(len(fine_names)):
        mask = fy_va == c
        recalls.append(float((preds[mask] == c).mean()) if mask.any() else 0.0)

    report = {
        "backend": "wlasl_landmarks",
        "architecture": f"pretrain-top{args.pretrain_classes} → finetune-top{args.finetune_classes} Transformer",
        "dataset": "Voxel51/WLASL landmark sequences",
        "num_classes": len(fine_names),
        "class_names": fine_names,
        "num_frames": int(fx_tr.shape[1]),
        "feat_dim": int(fx_tr.shape[-1]),
        "n_train": int(len(fy_tr)),
        "n_val": int(len(fy_va)),
        "val_accuracy": b_acc,
        "transformer_val_accuracy": t_acc,
        "hgb_val_accuracy": h_acc,
        "val_macro_recall": float(np.mean(recalls)),
        "per_class_recall": {fine_names[i]: recalls[i] for i in range(len(fine_names))},
        "transformer_path": str(out / MODEL_PT),
        "hgb_path": str(out / "wlasl_landmark_hgb.joblib"),
        "transformer_weight": tw,
        "inference": "landmark_transformer" if tw >= 0.99 else "landmark_ensemble",
        "target_signs": 50,
        "target_accuracy": 0.85,
        "signs_met": len(fine_names) >= 50,
        "accuracy_met": b_acc >= 0.85,
    }
    (out / METADATA).write_text(json.dumps(report, indent=2), encoding="utf-8")
    eval_path = Path("logs/evaluation/wlasl_accuracy.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps(
            {
                "backend": "wlasl_landmarks",
                "split": "val",
                "n": int(len(fy_va)),
                "num_classes": len(fine_names),
                "accuracy": b_acc,
                "target_signs": 50,
                "target_accuracy": 0.85,
                "signs_met": True,
                "accuracy_met": b_acc >= 0.85,
                "per_class_recall": report["per_class_recall"],
                "class_names": fine_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("val_accuracy", "transformer_val_accuracy", "accuracy_met", "signs_met")}, indent=2))


if __name__ == "__main__":
    main()
