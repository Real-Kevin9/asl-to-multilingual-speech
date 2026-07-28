# MediaPipe Hand Landmark Setup Guide

## Current Status

This project uses MediaPipe 0.10.35 (Tasks API) for hand landmark detection in ASL preprocessing.

**Issue**: MediaPipe Tasks API requires an external TFLite model file (`hand_landmarker.tflite`), which is not bundled with the package and cannot be automatically downloaded from this environment.

## Solution

You have two options:

### Option A: Download and Use the Model (Recommended)

1. **Download the model locally** (on your machine with internet access):
   - Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
   - Download `hand_landmarker.tflite` (the float16 or float32 version)

2. **Place it in the project**:
   ```bash
   cp ~/Downloads/hand_landmarker.tflite models/recognition/hand_landmarker.tflite
   ```

3. **Run preprocessing**:
   ```bash
   ./.venv/bin/python scripts/preprocess_asl.py \
     --input-dir data/raw/asl_alphabet/asl_alphabet_train \
     --output-dir data/processed \
     --model-path models/recognition/hand_landmarker.tflite
   ```

### Option B: Alternative (No Model Required)

If you can't access the model file, the preprocessing will gracefully fall back to returning zero vectors with a warning. This allows you to:
- Test the data pipeline infrastructure
- Run training (with dummy features for now)
- Later replace with real landmarks once model is available

Run without model path:
```bash
./.venv/bin/python scripts/preprocess_asl.py
```

## Implementation Details

### MediaPipe Tasks API Usage

The preprocessing code (`modules/recognition/preprocess.py`) implements:

```python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Initialize with model
options = HandLandmarkerOptions(
    base_options=base_options_lib.BaseOptions(model_asset_path=model_path),
    running_mode=VisionTaskRunningMode.IMAGE,
)
landmarker = HandLandmarker.create_from_options(options)

# Detect landmarks
image = image_lib.Image.create_from_file(image_path)
result = landmarker.detect(image)
```

### Feature Format

- Each image produces 21 hand landmarks (MediaPipe Hand)
- Each landmark has (x, y, z) coordinates
- Total: **63 floats per image** (21 × 3)
- Stored as numpy float32

## Python Environment

**Critical**: Always use the virtual environment:
```bash
./.venv/bin/python  # correct
python               # incorrect (system Python)
```

The project depends on specific package versions in `.venv`:
- mediapipe==0.10.35
- opencv-python==4.13.0.92
- numpy, scikit-learn, etc.

## Troubleshooting

### Import Error: "mediapipe.tasks not available"
- Ensure you're using `./.venv/bin/python`
- Verify `.venv` exists and is activated

### FileNotFoundError: Model asset not found
- Download `hand_landmarker.tflite` and save to `models/recognition/`
- Pass full path via `--model-path` argument

### Preprocessing returns all zeros
- Model file is not found (check file path)
- MediaPipe Tasks API issue (check terminal output for details)

## Next Steps

1. Obtain the `hand_landmarker.tflite` model
2. Place in `models/recognition/hand_landmarker.tflite`
3. Run preprocessing script
4. Verify non-zero feature vectors: `python -c "import numpy as np; f = np.load('data/processed/preprocessed_features.npy'); print(np.count_nonzero(f), '/', f.size)"`
5. Continue to model training

## References

- MediaPipe Tasks Documentation: https://developers.google.com/mediapipe/solutions
- Hand Landmarker Guide: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- Model Downloads: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#get_help
