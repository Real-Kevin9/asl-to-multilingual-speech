#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 generate_figures.py 2>/dev/null || true
# Real UI screenshots: figures/working_web-app.png, figures/alphabet-tryon-web-app.png
pdflatex -interaction=nonstopmode final_progress_review.tex
bibtex final_progress_review
pdflatex -interaction=nonstopmode final_progress_review.tex
pdflatex -interaction=nonstopmode final_progress_review.tex
echo "Built: final_progress_review.pdf"
