# Environment Setup & Preprocessing Status

## Summary

✅ **Environment Validated**: `.venv` Python environment has all required dependencies
✅ **Preprocessing Infrastructure**: Scripts work and successfully process ASL dataset  
⚠️ **MediaPipe Model**: Requires external TFLite file for real hand landmark detection

## What Works

### Virtual Environment (`.venv`)
- Python 3.12.3 with correct package versions
- `opencv-python==4.13.0.92` ✓
- `mediapipe==0.10.35` ✓
- `scikit-learn` ✓
- `numpy` ✓

### Data Processing
- Loaded 204,924 images from ASL alphabet dataset
- Successfully extracted class labels
- Generated feature vectors (shape: 204,924 × 63)
- Saved to `data/processed/preprocessed_features.npy`

### Current Pipeline Status
```
data/raw/asl_alphabet/asl_alphabet_train/
  ├── A/
  ├── B/
  ...
  └── Z/
       └── [.jpg images]
          ↓ preprocessing script
  ├── preprocessed_features.npy (204,924 × 63 float32)
  └── labels.txt (class names)
```

## Next Steps

### Critical: Get Hand Landmarker Model
The preprocessing currently falls back to **zero vectors** (all features are 0.0) because no TFLite model is available.

**To get real hand landmarks:**

1. **Download model** (requires internet access on your machine)
   - Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
   - Download: `hand_landmarker.tflite` (float16 or float32 version)

2. **Copy to project**
   ```bash
   cp ~/Downloads/hand_landmarker.tflite models/recognition/hand_landmarker.tflite
   ```

3. **Re-run preprocessing** with model
   ```bash
   ./.venv/bin/python scripts/preprocess_asl.py \
     --model-path models/recognition/hand_landmarker.tflite
   ```

4. **Verify non-zero features**
   ```bash
   python3 -c "import numpy as np; f = np.load('data/processed/preprocessed_features.npy'); print(f'Non-zero: {np.count_nonzero(f)}/{f.size}')"
   ```

### Current Limitations

**Without MediaPipe Model**:
- ✓ Pipeline structure works
- ✓ Data loading works
- ✓ Label extraction works
- ✗ All feature vectors are zeros (no hand landmarks)
- → Training models with zero vectors is not meaningful

**Workaround for Testing**:
If model is unavailable and you want to proceed:
1. Generate synthetic hand landmarks for testing
2. Train model with dummy features (verifies training code)
3. Later replace features when model is available

## Architecture

### Preprocessing (`modules/recognition/preprocess.py`)

**API**:
```python
from modules.recognition.preprocess import preprocess_dataset

features, labels = preprocess_dataset(
    root_dir='data/raw/asl_alphabet/asl_alphabet_train',
    model_asset_path='models/recognition/hand_landmarker.tflite'  # Optional
)
```

**Backends**:
- MediaPipe Tasks (requires TFLite model) → 21 landmarks × 3 coords = 63 floats per image
- Fallback: returns zero vectors with warning

### Key Dependencies
- **MediaPipe 0.10.35**: `mediapipe.tasks.python.vision.HandLandmarker`
- **OpenCV 4.13.0**: Image reading and processing
- **NumPy**: Feature storage and manipulation

## Instructions for Different Scenarios

### Scenario A: Have the Model File
```bash
# Copy model to project
cp ~/Downloads/hand_landmarker.tflite models/recognition/

# Run preprocessing with model
./.venv/bin/python scripts/preprocess_asl.py --model-path models/recognition/hand_landmarker.tflite

# Verify
python3 -c "import numpy as np; f = np.load('data/processed/preprocessed_features.npy'); print(np.count_nonzero(f), '/', f.size)"
```

### Scenario B: Testing Without Model
```bash
# Current preprocessing (creates zero vectors)
./.venv/bin/python scripts/preprocess_asl.py

# Can proceed with model training to verify pipeline
# Later replace features when model is obtained
```

### Scenario C: Use Alternative Hand Detection
If MediaPipe model is not available and you need real landmarks:
- Consider: OpenCV hand gesture detection
- Note: Less accurate than MediaPipe, but no model download needed
- File: Would modify `modules/recognition/preprocess.py`

## Environment Checklist

- [x] Python 3.12.3 in `.venv`
- [x] MediaPipe 0.10.35 installed
- [x] OpenCV installed
- [x] NumPy, scikit-learn installed
- [x] Preprocessing scripts run without errors
- [x] Dataset loads successfully
- [ ] **TODO**: Obtain hand_landmarker.tflite model
- [ ] **TODO**: Run preprocessing with model
- [ ] **TODO**: Validate non-zero features
- [ ] **TODO**: Proceed to model training

## Files Modified/Created

- `modules/recognition/preprocess.py`: Updated to use MediaPipe Tasks API with fallback
- `scripts/preprocess_asl.py`: Updated with optional model path and no auto-download
- `MEDIAPIPE_SETUP.md`: Setup guide for MediaPipe hand landmarks
- `data/processed/`: Output of preprocessing (currently with zero vectors)

## Troubleshooting

**Issue**: `ImportError: mediapipe.tasks not available`
- **Fix**: Use `./.venv/bin/python` (not system Python)

**Issue**: Features are all zeros
- **Fix**: Provide TFLite model file via `--model-path`

**Issue**: Model file not found
- **Fix**: Download from MediaPipe site and save to `models/recognition/hand_landmarker.tflite`

**Issue**: Script hangs or is very slow
- **Fix**: MediaPipe Tasks initialization and hand detection is computationally expensive; preprocessing ~200k images may take hours

## References

- MediaPipe Solutions: https://developers.google.com/mediapipe/solutions
- Hand Landmarker: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- Project README: [README.md](README.md)
- MediaPipe Setup: [MEDIAPIPE_SETUP.md](MEDIAPIPE_SETUP.md)
