"""Fine-tune t5-small on ASLG-PC12 gloss → English pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from modules.nlp.data import TASK_PREFIX, format_source, load_jsonl, prepare_splits

DEFAULT_MODEL_NAME = "t5-small"
DEFAULT_OUTPUT_DIR = "models/nlp/t5_gloss_en"


class GlossT5Trainer:
    """Thin wrapper around Hugging Face ``Seq2SeqTrainer`` for gloss→English."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def train(
        self,
        train_pairs: Sequence[Dict[str, str]],
        val_pairs: Sequence[Dict[str, str]],
        epochs: float = 2.0,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        max_source_length: int = 64,
        max_target_length: int = 64,
        seed: int = 42,
    ) -> Dict[str, object]:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        def encode(pairs: Sequence[Dict[str, str]]):
            sources = [format_source(p["gloss"]) for p in pairs]
            targets = [p["text"] for p in pairs]
            model_inputs = tokenizer(
                sources,
                max_length=max_source_length,
                truncation=True,
                padding=False,
            )
            labels = tokenizer(
                text_target=targets,
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return [
                {
                    "input_ids": model_inputs["input_ids"][i],
                    "attention_mask": model_inputs["attention_mask"][i],
                    "labels": model_inputs["labels"][i],
                }
                for i in range(len(pairs))
            ]

        from datasets import Dataset

        train_ds = Dataset.from_list(encode(train_pairs))
        val_ds = Dataset.from_list(encode(val_pairs))

        use_fp16 = bool(torch.cuda.is_available())
        args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            logging_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,
            seed=seed,
            report_to=[],
        )

        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        meta = {
            "architecture": "t5-small gloss→English (ASLG-PC12)",
            "base_model": self.model_name,
            "task_prefix": TASK_PREFIX,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "train_loss": float(train_result.training_loss),
            "output_dir": str(self.output_dir),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return meta


def train_from_processed(
    processed_dir: str = "data/processed/aslg_pc12",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    max_train: int = 20000,
    max_val: int = 2000,
    epochs: float = 2.0,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> Dict[str, object]:
    """Prepare capped splits (if needed) and run fine-tuning."""
    all_path = Path(processed_dir) / "all.jsonl"
    train_path = Path(processed_dir) / "train.jsonl"
    val_path = Path(processed_dir) / "val.jsonl"

    if not all_path.exists():
        raise FileNotFoundError(
            f"{all_path} missing. Run scripts/download_aslg_pc12.py first."
        )

    if not train_path.exists() or not val_path.exists():
        train, val, _, _ = prepare_splits(
            processed_dir=processed_dir,
            max_train=max_train,
            max_val=max_val,
            seed=seed,
        )
    else:
        train = load_jsonl(train_path)[:max_train]
        val = load_jsonl(val_path)[:max_val]

    trainer = GlossT5Trainer(model_name=model_name, output_dir=output_dir)
    return trainer.train(
        train,
        val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
