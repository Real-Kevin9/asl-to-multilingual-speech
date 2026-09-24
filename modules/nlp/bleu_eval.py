"""BLEU helpers for gloss→English evaluation."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU (0–100). Falls back to a tiny n-gram BLEU if sacrebleu is absent."""
    hyps = [h if h is not None else "" for h in hypotheses]
    refs = [r if r is not None else "" for r in references]
    if len(hyps) != len(refs):
        raise ValueError(f"hypothesis/reference length mismatch: {len(hyps)} vs {len(refs)}")
    if not hyps:
        return 0.0

    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(hyps, [refs], force=True).score)
    except Exception:
        return _simple_corpus_bleu(hyps, refs)


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _simple_corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Lightweight corpus BLEU-4 used only when sacrebleu is unavailable."""
    import math
    from collections import Counter

    clipped = [0] * 4
    total = [0] * 4
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        ht = hyp.lower().split()
        rt = ref.lower().split()
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, 5):
            hc = Counter(_ngrams(ht, n))
            rc = Counter(_ngrams(rt, n))
            clipped[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[n - 1] += max(sum(hc.values()), 0)

    precisions = []
    for n in range(4):
        if total[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped[n] / total[n])

    if hyp_len == 0:
        return 0.0
    if min(precisions) == 0.0:
        return 0.0

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    return float(score * 100.0)


def mean_length(texts: Iterable[str]) -> float:
    items = list(texts)
    if not items:
        return 0.0
    return sum(len(t.split()) for t in items) / len(items)
