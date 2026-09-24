"""Generate figures for Final Progress Review report."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Interim Report §3.2 objectives (measurable metrics only)
labels = [
    "Obj 1\nAlphabet\naccuracy",
    "Obj 1\nWLASL\naccuracy",
    "Obj 2\nNLP BLEU",
    "Obj 3\nLatency\n(budget used)",
    "Obj 5\nMetrics\ndone",
]
targets = [85.0, 85.0, 50.0, 100.0, 100.0]
# Latency: 814ms/1500ms — met, shown as 100% attainment
display_achieved = [93.7, 60.7, 57.6, 100.0, 100.0]
colors = ["#2ecc71", "#e74c3c", "#2ecc71", "#2ecc71", "#3498db"]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.bar(x - width / 2, targets, width, label="Target (%)", color="#bdc3c7", edgecolor="white")
ax.bar(x + width / 2, display_achieved, width, label="Achieved (%)", color=colors, edgecolor="white")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Interim Report objectives: measured performance vs targets")
fig.tight_layout()
fig.savefig(OUT / "objectives_comparison.pdf", bbox_inches="tight")
fig.savefig(OUT / "objectives_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# WLASL accuracy progression
stages = ["RGB v1", "RGB v2", "Landmarks", "Boost ensemble"]
acc = [16.2, 27.4, 53.8, 60.7]
fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(stages, acc, marker="o", color="#3498db", linewidth=2, markersize=8)
ax.axhline(85, color="#e74c3c", linestyle="--", linewidth=1.2, label="Interim target (85%)")
ax.set_ylabel("Holdout accuracy (%)")
ax.set_ylim(0, 95)
ax.legend(fontsize=8)
ax.set_title("WLASL word recognition improvement (top-50 glosses)")
for i, v in enumerate(acc):
    ax.annotate(f"{v}%", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "wlasl_progression.pdf", bbox_inches="tight")
fig.savefig(OUT / "wlasl_progression.png", dpi=150, bbox_inches="tight")
plt.close()

print("Wrote figures to", OUT)
