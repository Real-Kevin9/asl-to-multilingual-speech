"""Inference for WLASL word-level recognition.

Prefers the landmark Transformer+HGB ensemble when artefacts exist (v3);
falls back to the MobileNetV2 feature + BiLSTM head pipeline (v2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from modules.recognition.wlasl_data import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_FRAMES,
    extract_clip_frames,
    frames_bgr_to_clip,
)
from modules.recognition.wlasl_trainer import (
    BACKBONE_FILENAME,
    HEAD_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    wlasl_custom_objects,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ImageLike = Union[str, np.ndarray]

LANDMARK_PT = "wlasl_landmark_transformer.pt"
LANDMARK_HGB = "wlasl_landmark_hgb.joblib"
LANDMARK_TGCN = "wlasl_landmark_tgcn.pt"
LANDMARK_ENSEMBLE = "wlasl_landmark_ensemble.pt"
LANDMARK_CNN = "wlasl_landmark_cnn.pt"
LANDMARK_KNN = "wlasl_landmark_knn.joblib"
R3D_PT = "wlasl_r3d18.pt"


class WLASLPredictor:
    """Classifies a short video / frame buffer as one WLASL gloss."""

    backend_name = "wlasl_video"

    def __init__(
        self,
        model_path: str = f"models/recognition/{MODEL_FILENAME}",
        metadata_path: Optional[str] = f"models/recognition/{METADATA_FILENAME}",
        head_path: Optional[str] = None,
        backbone_path: Optional[str] = None,
        min_confidence: float = 0.0,
        prefer_landmarks: bool = True,
    ):
        meta_file = Path(metadata_path) if metadata_path else None
        meta = {}
        if meta_file and meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))

        self.class_names: List[str] = list(meta.get("class_names", []))
        self.num_frames = int(meta.get("num_frames", DEFAULT_FRAMES))
        self.frame_size = int(meta.get("frame_size", DEFAULT_FRAME_SIZE))
        self.min_confidence = float(min_confidence)
        self.meta = meta

        base = Path(model_path).parent if model_path else Path("models/recognition")
        pt_file = base / LANDMARK_PT
        hgb_file = base / LANDMARK_HGB
        tgcn_file = base / LANDMARK_TGCN
        r3d_file = base / R3D_PT

        self._mode = "rgb"
        self.head = None
        self.backbone = None
        self.model = None
        self._torch_model = None
        self._torch_models: List[object] = []
        self._tgcn_model = None
        self._tgcn_models: List[object] = []
        self._hgb = None
        self._cnn_model = None
        self._knn = None
        self._v3_expert_dir: Optional[Path] = None
        self._v3_runtime = None
        self._chase_v4 = False
        self._r3d = None
        self._landmark_extractor = None
        self._feat_dim = int(meta.get("feat_dim", 225))
        self._use_velocity = bool(meta.get("velocity", False))
        self._use_geometry = bool(meta.get("geometry", False))

        inference = str(meta.get("inference") or "")
        if prefer_landmarks and inference == "r3d18" and r3d_file.exists():
            self._load_r3d(r3d_file, meta)
            self._mode = "r3d"
            self.backend_name = "wlasl_r3d"
        elif prefer_landmarks and pt_file.exists() and hgb_file.exists():
            self._load_landmark_ensemble(pt_file, hgb_file, meta, tgcn_file if tgcn_file.exists() else None)
            self._mode = "landmarks"
            self.backend_name = "wlasl_landmarks"
        elif prefer_landmarks and pt_file.exists():
            self._load_landmark_ensemble(
                pt_file,
                hgb_file if hgb_file.exists() else None,
                meta,
                tgcn_file if tgcn_file.exists() else None,
            )
            self._mode = "landmarks"
            self.backend_name = "wlasl_landmarks"
        else:
            self._load_rgb_pipeline(
                model_path=model_path,
                head_path=head_path,
                backbone_path=backbone_path,
                meta=meta,
                base=base,
            )

        if not self.class_names:
            n = self._num_classes()
            self.class_names = [str(i) for i in range(n)]

    def _num_classes(self) -> int:
        if self._torch_model is not None:
            return int(self._torch_model.head[-1].out_features)
        if self.head is not None:
            return int(self.head.output_shape[-1])
        if self.model is not None:
            return int(self.model.output_shape[-1])
        return len(self.class_names)

    def _load_landmark_ensemble(
        self,
        pt_file: Path,
        hgb_file: Optional[Path],
        meta: dict,
        tgcn_file: Optional[Path] = None,
    ) -> None:
        import torch

        from modules.recognition.wlasl_seq_trainer import _build_torch_model

        base = pt_file.parent
        ensemble_file = base / LANDMARK_ENSEMBLE
        seed_states: List[dict] = []
        tgcn_seed_states: List[dict] = []
        ens_meta: dict = {}
        if ensemble_file.exists():
            ens = torch.load(ensemble_file, map_location="cpu", weights_only=False)
            ens_meta = ens
            seed_states = list(ens.get("seeds") or [])
            tgcn_seed_states = list(ens.get("tgcn_seeds") or [])
            if ens.get("blend_weights"):
                self.meta = {**self.meta, "blend_weights": ens["blend_weights"]}
            self._use_geometry = bool(ens.get("geometry") or meta.get("geometry"))

        if seed_states:
            ckpt = seed_states[0]
            self.class_names = list(ckpt.get("class_names") or meta.get("class_names") or [])
            self.num_frames = int(ckpt.get("num_frames") or meta.get("num_frames") or 32)
            self._feat_dim = int(ckpt.get("feat_dim") or meta.get("feat_dim") or 225)
            self._use_velocity = bool(ckpt.get("velocity") or meta.get("velocity"))
            self._use_geometry = bool(
                ckpt.get("geometry") or meta.get("geometry") or self._feat_dim > 450
            )
            for state in seed_states:
                model = _build_torch_model(
                    num_classes=int(state["num_classes"]),
                    feat_dim=self._feat_dim,
                    d_model=int(state.get("d_model", 128)),
                    nhead=int(state.get("nhead", 4)),
                    layers=int(state.get("layers", 3)),
                )
                model.load_state_dict(state["state_dict"])
                model.eval()
                self._torch_models.append(model)
            self._torch_model = self._torch_models[0]
            self._chase_v4 = bool(
                self._use_geometry or self._feat_dim > 450 or ens_meta.get("has_cnn")
            )
        else:
            ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
            self.class_names = list(ckpt.get("class_names") or meta.get("class_names") or [])
            self.num_frames = int(ckpt.get("num_frames") or meta.get("num_frames") or 32)
            self._feat_dim = int(ckpt.get("feat_dim") or meta.get("feat_dim") or 225)
            self._use_velocity = bool(ckpt.get("velocity") or meta.get("velocity"))
            model = _build_torch_model(
                num_classes=int(ckpt["num_classes"]),
                feat_dim=self._feat_dim,
                d_model=int(ckpt.get("d_model", 128)),
                nhead=int(ckpt.get("nhead", 4)),
                layers=int(ckpt.get("layers", 3)),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._torch_model = model
            self._torch_models = [model]
        self._hgb = None
        if hgb_file is not None and Path(hgb_file).exists():
            import joblib

            self._hgb = joblib.load(hgb_file)
        self._tgcn_model = None
        self._tgcn_models = []
        from modules.recognition.wlasl_boost_trainer import _load_tgcn_cls

        def _build_tgcn(tckpt: dict):
            GCN = _load_tgcn_cls()
            tgcn = GCN(
                input_feature=int(tckpt["t_dim"]),
                hidden_feature=int(tckpt.get("hidden", 96)),
                num_class=int(tckpt["num_classes"]),
                p_dropout=float(tckpt.get("p_dropout", 0.3)),
                num_stage=int(tckpt.get("num_stage", 12)),
            )
            tgcn.load_state_dict(tckpt["state_dict"])
            tgcn.eval()
            return tgcn

        if tgcn_seed_states:
            for tstate in tgcn_seed_states:
                self._tgcn_models.append(_build_tgcn(tstate))
            self._tgcn_model = self._tgcn_models[0]
        elif tgcn_file is not None and Path(tgcn_file).exists():
            tckpt = torch.load(tgcn_file, map_location="cpu", weights_only=False)
            self._tgcn_model = _build_tgcn(tckpt)
            self._tgcn_models = [self._tgcn_model]

        cnn_file = base / LANDMARK_CNN
        knn_file = base / LANDMARK_KNN
        if self._chase_v4 and cnn_file.exists():
            from modules.recognition.wlasl_chase_v4 import build_cnn_lstm

            cckpt = torch.load(cnn_file, map_location="cpu", weights_only=False)
            cnn = build_cnn_lstm(int(cckpt["num_classes"]), int(cckpt["feat_dim"]))()
            cnn.load_state_dict(cckpt["state_dict"])
            cnn.eval()
            self._cnn_model = cnn
        if self._chase_v4 and knn_file.exists():
            import joblib

            self._knn = joblib.load(knn_file)
        v3_dir = meta.get("v3_expert_dir")
        if v3_dir:
            self._v3_expert_dir = Path(v3_dir)
        else:
            candidate = base / "experts" / "v3"
            if (candidate / "wlasl_landmark_ensemble.pt").exists():
                self._v3_expert_dir = candidate
        if self._chase_v4 and self._v3_expert_dir is not None:
            from modules.recognition.wlasl_chase_v4 import FrozenV3ExpertRuntime

            self._v3_runtime = FrozenV3ExpertRuntime.load(
                self._v3_expert_dir, self.class_names
            )

    def _load_r3d(self, r3d_file: Path, meta: dict) -> None:
        import torch
        from torch import nn
        from torchvision.models.video import r3d_18

        ckpt = torch.load(r3d_file, map_location="cpu", weights_only=False)
        self.class_names = list(ckpt.get("class_names") or meta.get("class_names") or [])
        self.num_frames = int(ckpt.get("num_frames") or 16)
        self.frame_size = int(ckpt.get("frame_size") or 112)
        model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, int(ckpt["num_classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._r3d = model

    def _load_rgb_pipeline(
        self,
        *,
        model_path: str,
        head_path: Optional[str],
        backbone_path: Optional[str],
        meta: dict,
        base: Path,
    ) -> None:
        from tensorflow import keras

        head_file = Path(head_path or meta.get("head_path") or (base / HEAD_FILENAME))
        backbone_file = Path(
            backbone_path or meta.get("backbone_path") or (base / BACKBONE_FILENAME)
        )
        load_kwargs = {
            "custom_objects": wlasl_custom_objects(),
            "safe_mode": False,
        }
        if head_file.exists() and backbone_file.exists():
            self.head = keras.models.load_model(head_file, **load_kwargs)
            self.backbone = keras.models.load_model(backbone_file)
            self.model = None
        else:
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"WLASL model not found: {model_file} "
                    f"(also missing landmark ensemble / head+backbone)"
                )
            self.model = keras.models.load_model(model_file, **load_kwargs)
            self.head = None
            self.backbone = None

    def _label(self, index: int) -> str:
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return str(index)

    def _ensure_landmark_extractor(self):
        if self._landmark_extractor is None:
            from modules.recognition.wlasl_landmarks import LandmarkExtractor

            # IMAGE mode per subsampled frame (matches offline cache extraction).
            self._landmark_extractor = LandmarkExtractor(video_mode=False)
        return self._landmark_extractor

    def _predict_probs_rgb(self, clip: np.ndarray) -> np.ndarray:
        if self.head is not None and self.backbone is not None:
            t = clip.shape[0]
            flat = clip.reshape(t, clip.shape[1], clip.shape[2], 3)
            feats = self.backbone.predict(flat, verbose=0)
            feats = np.expand_dims(feats, axis=0)
            return self.head.predict(feats, verbose=0)[0]
        batch = np.expand_dims(clip, axis=0)
        return self.model.predict(batch, verbose=0)[0]

    def _predict_probs_landmarks(self, seq: np.ndarray) -> np.ndarray:
        import torch

        from modules.recognition.wlasl_boost_trainer import batch_to_joints55, with_velocity
        from modules.recognition.wlasl_seq_trainer import augment_seq, temporal_stats

        seq = seq.astype(np.float32)
        if self._chase_v4:
            from modules.recognition.wlasl_chase_v4 import predict_chase_v4_probs

            blend = self.meta.get("blend_weights") or {}
            return predict_chase_v4_probs(
                seq,
                class_names=self.class_names,
                blend_weights=blend,
                torch_models=self._torch_models,
                tgcn_models=self._tgcn_models or ([self._tgcn_model] if self._tgcn_model else []),
                cnn_model=self._cnn_model,
                hgb=self._hgb,
                knn=self._knn,
                v3_expert_dir=self._v3_expert_dir,
                v3_runtime=self._v3_runtime,
            )
        if self._use_velocity or self._feat_dim == seq.shape[-1] * 2:
            seq = with_velocity(seq)

        blend = self.meta.get("blend_weights") or {}
        wt = float(blend.get("transformer", self.meta.get("transformer_weight", 1.0)))
        wg = float(blend.get("tgcn", 0.0))
        wh = float(blend.get("hgb", max(0.0, 1.0 - wt - wg)))

        xs = [seq]
        rng = np.random.default_rng(0)
        for _ in range(4):
            xs.append(augment_seq(seq, rng))
        batch = np.stack(xs, axis=0)
        tta_probs = []
        with torch.no_grad():
            for model in self._torch_models:
                logits = model(torch.tensor(batch, dtype=torch.float32))
                tta_probs.append(torch.softmax(logits, dim=-1).numpy().mean(axis=0))
        t_probs = np.mean(tta_probs, axis=0) if tta_probs else np.zeros(len(self.class_names))

        probs = wt * t_probs
        total_w = wt

        tgcn_models = self._tgcn_models or ([self._tgcn_model] if self._tgcn_model is not None else [])
        if tgcn_models and wg > 0:
            raw = seq[:, :225] if seq.shape[-1] >= 225 else seq
            j = batch_to_joints55(np.expand_dims(raw, 0))
            with torch.no_grad():
                g_list = [
                    torch.softmax(m(torch.tensor(j, dtype=torch.float32)), dim=-1).numpy()[0]
                    for m in tgcn_models
                ]
                g_probs = np.mean(g_list, axis=0)
            probs = probs + wg * g_probs
            total_w += wg

        if self._hgb is not None and wh > 0.01:
            raw = seq[:, :225] if seq.shape[-1] >= 225 else seq
            h_probs_raw = self._hgb.predict_proba(
                np.expand_dims(temporal_stats(raw), axis=0)
            )[0]
            classes = getattr(self._hgb, "classes_", None)
            if classes is None and hasattr(self._hgb, "named_steps"):
                classes = self._hgb.named_steps["hgb"].classes_
            h_probs = np.zeros_like(t_probs)
            if classes is not None:
                for i, c in enumerate(classes):
                    h_probs[int(c)] = h_probs_raw[i]
            else:
                h_probs = h_probs_raw
            probs = probs + wh * h_probs
            total_w += wh

        if total_w <= 0:
            return t_probs
        return probs / total_w

    def _result_from_probs(self, probs: np.ndarray) -> dict:
        index = int(np.argmax(probs))
        conf = float(probs[index])
        label = self._label(index)
        top5 = [
            {"label": self._label(i), "confidence": float(probs[i])}
            for i in np.argsort(probs)[::-1][:5]
        ]
        if conf < self.min_confidence:
            return {
                "pred_index": -1,
                "label": "unknown",
                "confidence": conf,
                "backend": self.backend_name,
                "rejected": True,
                "raw_label": label,
                "top5": top5,
            }
        return {
            "pred_index": index,
            "label": label,
            "confidence": conf,
            "backend": self.backend_name,
            "top5": top5,
        }

    def _predict_probs_r3d(self, clip_rgb_0_255: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        # clip: (T,H,W,3) -> (1,C,T,112,112)
        t = clip_rgb_0_255.shape[0]
        idxs = np.linspace(0, t - 1, self.num_frames).round().astype(int)
        frames = clip_rgb_0_255[idxs]
        resized = np.stack(
            [
                cv2.resize(np.clip(f, 0, 255).astype(np.uint8), (self.frame_size, self.frame_size))
                for f in frames
            ],
            axis=0,
        ).astype(np.float32) / 255.0
        x = np.transpose(resized, (3, 0, 1, 2))  # C,T,H,W
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1, 1)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            logits = self._r3d(torch.tensor(x[None], dtype=torch.float32))
            return torch.softmax(logits, dim=-1).numpy()[0]

    def predict_clip(
        self,
        video_path: Optional[str] = None,
        frames_bgr: Optional[Sequence[np.ndarray]] = None,
        clip_array: Optional[np.ndarray] = None,
    ) -> dict:
        """Predict a single word-level gloss from a clip."""
        if self._mode == "r3d":
            if frames_bgr is not None:
                if cv2 is None:
                    raise ImportError("opencv required")
                clip = np.stack(
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) for f in frames_bgr],
                    axis=0,
                )
            elif video_path is not None:
                clip = extract_clip_frames(
                    video_path,
                    num_frames=max(self.num_frames, 24),
                    frame_size=max(self.frame_size, 160),
                    use_center_crop=True,
                )
                if clip is None:
                    return {
                        "pred_index": -1,
                        "label": "unknown",
                        "confidence": 0.0,
                        "backend": self.backend_name,
                        "error": f"Could not read video {video_path}",
                    }
            elif clip_array is not None:
                clip = np.asarray(clip_array, dtype=np.float32)
            else:
                raise ValueError("Provide video_path, frames_bgr, or clip_array")
            return self._result_from_probs(self._predict_probs_r3d(clip))

        if self._mode == "landmarks":
            from modules.recognition.wlasl_landmarks import (
                extract_video_landmarks,
                frames_bgr_to_landmark_seq,
            )

            extractor = self._ensure_landmark_extractor()
            if frames_bgr is not None:
                seq = frames_bgr_to_landmark_seq(
                    frames_bgr, extractor, num_frames=self.num_frames
                )
            elif video_path is not None:
                seq = extract_video_landmarks(
                    video_path, extractor, num_frames=self.num_frames
                )
                if seq is None:
                    return {
                        "pred_index": -1,
                        "label": "unknown",
                        "confidence": 0.0,
                        "backend": self.backend_name,
                        "error": f"Could not read video {video_path}",
                    }
            elif clip_array is not None:
                arr = np.asarray(clip_array, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[-1] in (225, 450, self._feat_dim):
                    seq = arr
                else:
                    # RGB clip provided — convert via temporary BGR frames if possible
                    if cv2 is None:
                        raise ValueError("clip_array RGB path needs opencv for landmark mode")
                    frames = [
                        cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        for f in arr
                    ]
                    seq = frames_bgr_to_landmark_seq(frames, extractor, num_frames=self.num_frames)
            else:
                raise ValueError("Provide video_path, frames_bgr, or clip_array")
            probs = self._predict_probs_landmarks(seq)
            return self._result_from_probs(probs)

        # RGB path
        if clip_array is not None:
            clip = np.asarray(clip_array, dtype=np.float32)
            if clip.shape[1] != self.frame_size or clip.shape[2] != self.frame_size:
                if cv2 is None:
                    raise ImportError("opencv is required to resize clips")
                clip = np.stack(
                    [
                        cv2.resize(f, (self.frame_size, self.frame_size)).astype(np.float32)
                        for f in clip
                    ],
                    axis=0,
                )
        elif frames_bgr is not None:
            clip = frames_bgr_to_clip(
                frames_bgr,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
            )
        elif video_path is not None:
            clip = extract_clip_frames(
                video_path,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                use_center_crop=True,
            )
            if clip is None:
                return {
                    "pred_index": -1,
                    "label": "unknown",
                    "confidence": 0.0,
                    "backend": self.backend_name,
                    "error": f"Could not read video {video_path}",
                }
        else:
            raise ValueError("Provide video_path, frames_bgr, or clip_array")

        if clip.shape[0] != self.num_frames:
            from modules.recognition.wlasl_data import sample_frame_indices

            idxs = sample_frame_indices(clip.shape[0], self.num_frames)
            clip = clip[idxs]

        probs = self._predict_probs_rgb(clip)
        return self._result_from_probs(probs)

    def predict_frame(self, **kwargs) -> dict:
        return {
            "pred_index": -1,
            "label": "nothing",
            "confidence": 0.0,
            "hand_detected": False,
            "backend": self.backend_name,
            "note": "Use predict_clip() for word-level WLASL recognition.",
        }

    def close(self) -> None:
        if self._landmark_extractor is not None:
            self._landmark_extractor.close()
            self._landmark_extractor = None
