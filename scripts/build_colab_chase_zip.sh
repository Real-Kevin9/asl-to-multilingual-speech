#!/usr/bin/env bash
# Rebuild wlasl_colab_chase.zip for Colab chase v3 (prefers landmarks_v4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/colab_wlasl_chase"
STAGE="$PKG/wlasl_colab_chase"
ZIP_OUT="$ROOT/wlasl_colab_chase.zip"

if [[ -f "$ROOT/data/processed/wlasl_landmarks_v4/metadata.json" ]]; then
  LM_SRC="$ROOT/data/processed/wlasl_landmarks_v4"
  LM_NAME=wlasl_landmarks_v4
else
  LM_SRC="$ROOT/data/processed/wlasl_landmarks_v3"
  LM_NAME=wlasl_landmarks_v3
  echo "WARN: v4 metadata missing — packaging $LM_NAME"
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/modules/recognition" "$STAGE/scripts" \
  "$STAGE/data/processed" "$STAGE/models/recognition/tgcn_wlasl" \
  "$STAGE/logs/evaluation"

printf '' > "$STAGE/modules/__init__.py"
printf '' > "$STAGE/modules/recognition/__init__.py"

cp "$ROOT/modules/recognition/wlasl_boost_trainer.py" \
   "$ROOT/modules/recognition/wlasl_seq_trainer.py" \
   "$ROOT/modules/recognition/wlasl_landmarks.py" \
   "$ROOT/modules/recognition/wlasl_predictor.py" \
   "$ROOT/modules/recognition/wlasl_data.py" \
   "$ROOT/modules/recognition/wlasl_trainer.py" \
   "$ROOT/modules/recognition/keras_env.py" \
   "$STAGE/modules/recognition/"

cp "$ROOT/scripts/train_wlasl_boost.py" "$ROOT/scripts/evaluate_wlasl.py" "$STAGE/scripts/"
cp "$ROOT/models/recognition/tgcn_wlasl/tgcn_model.py" \
   "$ROOT/models/recognition/tgcn_wlasl/configs.py" \
   "$ROOT/models/recognition/tgcn_wlasl/README.md" \
   "$STAGE/models/recognition/tgcn_wlasl/"

cp -a "$LM_SRC" "$STAGE/data/processed/$LM_NAME"
# Keep v3 as fallback if packaging v4
if [[ "$LM_NAME" == "wlasl_landmarks_v4" && -d "$ROOT/data/processed/wlasl_landmarks_v3" ]]; then
  cp -a "$ROOT/data/processed/wlasl_landmarks_v3" "$STAGE/data/processed/wlasl_landmarks_v3"
fi

cp "$PKG/WLASL_Colab_Chase.ipynb" "$PKG/README.md" "$STAGE/"
cat > "$STAGE/requirements-colab.txt" <<'EOF'
joblib
scikit-learn
numpy
EOF
cat > "$STAGE/BASELINE.json" <<'EOF'
{
  "live_val_accuracy": 0.6495726495726496,
  "num_classes": 50,
  "note": "Beat this with chase_colab_v3 outputs (preferably on wlasl_landmarks_v4)"
}
EOF

cd "$PKG"
rm -f "$ZIP_OUT"
zip -r -q "$ZIP_OUT" wlasl_colab_chase -x '*/__pycache__/*' '*.pyc'
cp -f "$STAGE/WLASL_Colab_Chase.ipynb" "$ROOT/WLASL_Colab_Chase_v3.ipynb"
ls -lh "$ZIP_OUT" "$ROOT/WLASL_Colab_Chase_v3.ipynb"
python3 - <<PY
import json
from pathlib import Path
meta = Path("$STAGE/data/processed/$LM_NAME/metadata.json")
m = json.loads(meta.read_text())
print("packaged", "$LM_NAME", "train_views=", m.get("train_views"), "n_train=", m.get("n_train"), "n_val=", m.get("n_val"))
PY
