"""Compare rule-based vs fine-tuned T5 BLEU on held-out ASLG-PC12 pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.nlp.bleu_eval import corpus_bleu, mean_length  # noqa: E402
from modules.nlp.correction import GrammarCorrector  # noqa: E402
from modules.nlp.data import load_jsonl  # noqa: E402
from modules.nlp.trainer import DEFAULT_OUTPUT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="BLEU evaluation for gloss→English NLP")
    parser.add_argument("--val-jsonl", default="data/processed/aslg_pc12/val.jsonl")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on evaluation pairs (faster smoke runs).")
    parser.add_argument("--output", default="logs/evaluation/nlp_bleu.json")
    args = parser.parse_args()

    pairs = load_jsonl(args.val_jsonl)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        print("No evaluation pairs found.")
        sys.exit(1)

    glosses = [p["gloss"] for p in pairs]
    references = [p["text"] for p in pairs]

    rules = GrammarCorrector(use_model=False)
    t0 = time.perf_counter()
    rule_hyps = [rules.correct_rule_based(g) for g in glosses]
    rule_ms = (time.perf_counter() - t0) * 1000 / len(glosses)
    rule_bleu = corpus_bleu(rule_hyps, references)

    t5 = GrammarCorrector(
        use_model=True,
        local_model_dir=args.model_dir,
        model_name=None,
    )
    if t5.backend == "rule_based":
        print(
            f"No fine-tuned model at {args.model_dir}. "
            "Train one with scripts/train_nlp_t5.py"
        )
        report = {
            "n": len(pairs),
            "rule_based": {
                "bleu": rule_bleu,
                "avg_ms": rule_ms,
                "avg_hyp_len": mean_length(rule_hyps),
            },
            "t5_gloss": None,
            "delta_bleu": None,
            "examples": [
                {
                    "gloss": glosses[i],
                    "reference": references[i],
                    "rule_based": rule_hyps[i],
                }
                for i in range(min(5, len(pairs)))
            ],
        }
    else:
        t0 = time.perf_counter()
        t5_hyps = [t5.correct(g) for g in glosses]
        t5_ms = (time.perf_counter() - t0) * 1000 / len(glosses)
        t5_bleu = corpus_bleu(t5_hyps, references)
        report = {
            "n": len(pairs),
            "backend": t5.backend,
            "rule_based": {
                "bleu": rule_bleu,
                "avg_ms": rule_ms,
                "avg_hyp_len": mean_length(rule_hyps),
            },
            "t5_gloss": {
                "bleu": t5_bleu,
                "avg_ms": t5_ms,
                "avg_hyp_len": mean_length(t5_hyps),
            },
            "delta_bleu": t5_bleu - rule_bleu,
            "examples": [
                {
                    "gloss": glosses[i],
                    "reference": references[i],
                    "rule_based": rule_hyps[i],
                    "t5_gloss": t5_hyps[i],
                }
                for i in range(min(5, len(pairs)))
            ],
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Pairs evaluated : {report['n']}")
    print(f"Rule-based BLEU : {rule_bleu:.2f}")
    if report.get("t5_gloss"):
        print(f"T5 gloss BLEU   : {report['t5_gloss']['bleu']:.2f}")
        print(f"Delta           : {report['delta_bleu']:+.2f}")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
