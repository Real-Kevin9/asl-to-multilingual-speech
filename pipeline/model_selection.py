"""Resolve the best trained artefact for every pipeline stage.

Each stage was trained and evaluated separately (Updates 1–7), so "run the good
models" previously meant remembering three different flags. This module reads the
model directory once and reports what will actually be loaded, which is also what
the demo prints on startup so a run can be tied to specific artefacts.

Stage preference:

* **recognition** — ``hybrid_user`` > ``hybrid`` > ``cnn_lstm`` > ``landmark_mlp``
  (see ``modules/recognition/registry``). Word-level ``wlasl_video`` is reported
  separately when its artefact exists.
* **nlp** — local fine-tuned T5 at ``models/nlp/t5_gloss_en`` when present,
  otherwise the offline rule-based corrector. ``auto`` never reaches for the
  pretrained model over the network.
* **emotion** — ``models/emotion/emotion_model.keras`` when present, otherwise a
  neutral fallback so the pipeline still completes.
* **tts** — always gTTS with the disk cache from Update 7.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.emotion.detector import DEFAULT_METADATA_PATH as EMOTION_METADATA
from modules.emotion.detector import DEFAULT_MODEL_PATH as EMOTION_MODEL
from modules.nlp.correction import DEFAULT_LOCAL_MODEL as NLP_LOCAL_MODEL
from modules.recognition.registry import available_backends, backend_model_path

DEFAULT_RECOGNITION_DIR = "models/recognition"
DEFAULT_EVAL_DIR = "logs/evaluation"

# Recognition metadata sits next to each model with a matching name.
_RECOGNITION_METADATA = {
    "hybrid_user": "hybrid_user_metadata.json",
    "hybrid": "hybrid_metadata.json",
    "cnn_lstm": "cnn_lstm_metadata.json",
    "landmark_mlp": None,
    "wlasl_video": "wlasl_video_metadata.json",
}

NLP_MODES = ("auto", "on", "off")


@dataclass
class StageSelection:
    """What a single stage will load, and how good it measured."""

    stage: str
    backend: str
    available: bool
    artefact: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class ModelSelection:
    recognition: StageSelection
    nlp: StageSelection
    emotion: StageSelection
    tts: StageSelection
    wlasl: Optional[StageSelection] = None

    @property
    def stages(self) -> List[StageSelection]:
        stages = [self.recognition, self.nlp, self.emotion, self.tts]
        if self.wlasl is not None:
            stages.append(self.wlasl)
        return stages

    def as_dict(self) -> Dict[str, Any]:
        return {stage.stage: asdict(stage) for stage in self.stages}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def nlp_local_model_available(model_dir: str = NLP_LOCAL_MODEL) -> bool:
    """True when a fine-tuned gloss→English T5 is on disk."""
    path = Path(model_dir)
    return path.exists() and (path / "config.json").exists()


def resolve_nlp_use_model(mode: str = "auto", model_dir: str = NLP_LOCAL_MODEL) -> bool:
    """Map an ``auto``/``on``/``off`` request to the pipeline's boolean flag."""
    if mode not in NLP_MODES:
        raise ValueError(f"Unknown NLP mode {mode!r}; expected one of {NLP_MODES}")
    if mode == "off":
        return False
    if mode == "on":
        return True
    return nlp_local_model_available(model_dir)


def _resolve_recognition(
    backend: str,
    model_dir: str,
    eval_dir: str = DEFAULT_EVAL_DIR,
) -> StageSelection:
    base = Path(model_dir)
    found = available_backends(model_dir)

    if backend == "auto":
        chosen = found[0] if found else None
        note = (
            "auto: " + " > ".join(found) if found
            else "No recogniser artefacts found; train one first."
        )
    else:
        chosen = backend
        note = "explicitly requested"
        if backend not in found:
            note = f"requested {backend!r} but its artefact is missing"

    if chosen is None:
        return StageSelection("recognition", "none", False, note=note)

    artefact = backend_model_path(chosen, model_dir)
    metadata_name = _RECOGNITION_METADATA.get(chosen)
    meta = _read_json(base / metadata_name) if metadata_name else None

    metrics: Dict[str, Any] = {}
    if meta:
        stage_metrics = meta.get("metrics") or {}
        if stage_metrics.get("val_accuracy") is not None:
            metrics["val_accuracy"] = stage_metrics["val_accuracy"]
        if stage_metrics.get("macro_f1") is not None:
            metrics["macro_f1"] = stage_metrics["macro_f1"]
        finetune = meta.get("user_finetune") or {}
        if finetune.get("after") is not None:
            metrics["user_holdout_accuracy"] = finetune["after"]

    # The evaluation report measures the same held-out images through the full
    # inference path, so prefer it over the figure the fine-tune recorded itself.
    user_report = _read_json(Path(eval_dir) / "user_data_accuracy.json") or {}
    measured = (user_report.get(chosen) or {}).get("accuracy")
    if isinstance(measured, (int, float)):
        metrics["user_holdout_accuracy"] = measured

    return StageSelection(
        stage="recognition",
        backend=chosen,
        available=artefact.exists(),
        artefact=str(artefact),
        metrics=metrics,
        note=note,
    )


def _resolve_nlp(mode: str, model_dir: str, eval_dir: str) -> StageSelection:
    local = nlp_local_model_available(model_dir)
    use_model = resolve_nlp_use_model(mode, model_dir)

    if use_model and local:
        backend, note = "t5_gloss", f"{mode}: fine-tuned T5 on disk"
    elif use_model:
        backend, note = "t5_pretrained", f"{mode}: no local T5, will try pretrained"
    else:
        backend = "rule_based"
        note = "off: rule-based requested" if mode == "off" else "auto: no local T5 found"

    metrics: Dict[str, Any] = {}
    bleu = _read_json(Path(eval_dir) / "nlp_bleu.json")
    if bleu:
        # The report keys its neural section by the backend that produced it, so
        # look that up rather than assuming a label.
        report_backend = bleu.get("backend") or "t5_finetuned"
        finetuned = bleu.get(report_backend) or bleu.get("t5_finetuned") or bleu.get("t5") or {}
        rules = bleu.get("rule_based") or {}
        if backend == "t5_gloss" and finetuned.get("bleu") is not None:
            metrics["bleu"] = finetuned["bleu"]
        if backend == "rule_based" and rules.get("bleu") is not None:
            metrics["bleu"] = rules["bleu"]

    return StageSelection(
        stage="nlp",
        backend=backend,
        available=local if backend == "t5_gloss" else True,
        artefact=str(Path(model_dir)) if local else None,
        metrics=metrics,
        note=note,
    )


def _resolve_emotion(
    model_path: Optional[str],
    metadata_path: str,
    eval_dir: str,
) -> StageSelection:
    path = Path(model_path or EMOTION_MODEL)
    exists = path.exists()

    metrics: Dict[str, Any] = {}
    meta = _read_json(Path(metadata_path))
    if meta and meta.get("val_accuracy") is not None:
        metrics["val_accuracy"] = meta["val_accuracy"]
    report = _read_json(Path(eval_dir) / "emotion_accuracy.json")
    if report and report.get("accuracy") is not None:
        metrics["test_accuracy"] = report["accuracy"]

    return StageSelection(
        stage="emotion",
        backend="cnn" if exists else "fallback_neutral",
        available=exists,
        artefact=str(path) if exists else None,
        metrics=metrics,
        note="trained CNN" if exists else "no model; every utterance stays neutral",
    )


def _resolve_wlasl(model_dir: str, eval_dir: str = DEFAULT_EVAL_DIR) -> StageSelection:
    path = Path(model_dir) / "wlasl_video_model.keras"
    landmark_pt = Path(model_dir) / "wlasl_landmark_transformer.pt"
    meta = _read_json(Path(model_dir) / "wlasl_video_metadata.json") or {}
    eval_report = _read_json(Path(eval_dir) / "wlasl_accuracy.json") or {}
    metrics: Dict[str, Any] = {}
    if meta.get("val_accuracy") is not None:
        metrics["val_accuracy"] = meta["val_accuracy"]
    if meta.get("transformer_val_accuracy") is not None:
        metrics["transformer_val_accuracy"] = meta["transformer_val_accuracy"]
    if meta.get("num_classes") is not None:
        metrics["num_classes"] = meta["num_classes"]
    if eval_report.get("accuracy") is not None:
        metrics["heldout_accuracy"] = eval_report["accuracy"]
    use_landmarks = landmark_pt.exists() and meta.get("inference", "").startswith("landmark")
    exists = landmark_pt.exists() or path.exists()
    backend = "wlasl_landmarks" if use_landmarks or landmark_pt.exists() else (
        "wlasl_video" if path.exists() else "none"
    )
    note = {
        "wlasl_landmarks": "word-level pose+hands Transformer (WLASL)",
        "wlasl_video": "word-level WLASL MobileNetV2+BiLSTM",
        "none": "not trained; run scripts/train_wlasl_v3.py or scripts/pretrain_finetune_wlasl.py",
    }[backend]
    return StageSelection(
        stage="wlasl",
        backend=backend,
        available=exists,
        artefact=str(landmark_pt if landmark_pt.exists() else path) if exists else None,
        metrics=metrics,
        note=note,
    )


def resolve_models(
    recognition_backend: str = "auto",
    nlp_mode: str = "auto",
    recognition_dir: str = DEFAULT_RECOGNITION_DIR,
    nlp_model_dir: str = NLP_LOCAL_MODEL,
    emotion_model_path: Optional[str] = None,
    emotion_metadata_path: str = EMOTION_METADATA,
    eval_dir: str = DEFAULT_EVAL_DIR,
    languages: tuple = ("en", "ne"),
) -> ModelSelection:
    """Inspect disk and report the artefact each stage will load."""
    return ModelSelection(
        recognition=_resolve_recognition(recognition_backend, recognition_dir, eval_dir),
        nlp=_resolve_nlp(nlp_mode, nlp_model_dir, eval_dir),
        emotion=_resolve_emotion(emotion_model_path, emotion_metadata_path, eval_dir),
        tts=StageSelection(
            stage="tts",
            backend="gtts_cached",
            available=True,
            artefact=None,
            metrics={},
            note="languages: " + ", ".join(languages),
        ),
        wlasl=_resolve_wlasl(recognition_dir, eval_dir),
    )


def _format_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float) and value <= 1.0 and "bleu" not in key:
            parts.append(f"{key}={value * 100:.1f}%")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "-"


def format_model_report(selection: ModelSelection) -> str:
    """Human-readable startup banner listing the resolved models."""
    lines = ["Models in use:"]
    for stage in selection.stages:
        flag = "ok " if stage.available else "!! "
        lines.append(
            f"  {flag}{stage.stage:<12} {stage.backend:<14} "
            f"{_format_metrics(stage.metrics):<40} ({stage.note})"
        )
    return "\n".join(lines)
