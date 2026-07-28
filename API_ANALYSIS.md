# MediaPipe API Analysis: Solutions vs Tasks

## Findings

### 1. Legacy Solutions API Availability
**Status**: ❌ **NOT AVAILABLE**

- `mediapipe.solutions.hands` does **not exist** in MediaPipe 0.10.35
- Tested all available PyPI versions (0.10.5 - 0.10.35)
- **All versions only have Tasks API**
- Solutions API was completely removed before 0.10.5 release

```python
# This fails in ALL available MediaPipe versions:
import mediapipe.solutions.hands  # ModuleNotFoundError
```

### 2. Tasks API Availability
**Status**: ✅ **AVAILABLE but requires model**

- `mediapipe.tasks.python.vision.HandLandmarker` exists in 0.10.35
- **Requires** an explicit model file (no automatic download)
- No default or built-in model included
- Error without model: `ValueError: ExternalFile must specify at least one of 'file_content', 'file_name', 'file_pointer_meta' or 'file_descriptor_meta'`

```python
# This works but needs model_asset_path:
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
options = HandLandmarkerOptions(
    base_options=base_options_lib.BaseOptions(model_asset_path="hand_landmarker.tflite")
)
```

## Migration Necessity: ✅ YES, WAS REQUIRED

The migration to Tasks API was **not optional** — it was **forced** by MediaPipe's design evolution:

| Aspect | Solutions (Legacy) | Tasks (Current) |
|--------|------------------|-----------------|
| **Availability** | Removed in all PyPI versions | Only option in 0.10.x |
| **Model requirement** | Bundled with package | Must provide externally |
| **API complexity** | Simpler, fewer options | More options, more config |
| **Update status** | No longer maintained | Active development |

## Recommended Approach

### Option A: Use Tasks API with External Model (RECOMMENDED)
**Pros**:
- ✅ Only available option using current MediaPipe
- ✅ No version downgrades needed
- ✅ Future-proof (Tasks API is actively developed)
- ✅ More flexible for different hand-related tasks

**Cons**:
- ❌ Requires 80-90 MB model file download
- ❌ Model not automatically available

**Steps**:
```bash
# 1. Download model (on your machine with internet)
# From: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

# 2. Copy to project
cp hand_landmarker.tflite models/recognition/

# 3. Run preprocessing
./.venv/bin/python scripts/preprocess_asl.py \
  --model-path models/recognition/hand_landmarker.tflite
```

### Option B: Find & Install Older MediaPipe (NOT RECOMMENDED)
**Why not**:
- ❌ MediaPipe 0.9.x not available on PyPI (0.10.5+ only)
- ❌ Would need to find external sources (unofficial, risky)
- ❌ Old version = no security updates, no bug fixes
- ❌ Incompatible with newer Python versions
- ❌ No guarantee Solutions API works with 3.12 Python

**Not worth the risk.**

### Option C: Use OpenCV + Manual Hand Detection (NOT RECOMMENDED)
**Why not**:
- ❌ OpenCV hand detection is basic (not MediaPipe quality)
- ❌ Would need to write custom detection logic
- ❌ Much lower accuracy than both Solutions and Tasks API
- ❌ Defeats the purpose of using MediaPipe

## Summary

| Factor | Current Status |
|--------|----------------|
| **MediaPipe version** | 0.10.35 (only available on PyPI) |
| **Legacy Solutions API** | ❌ Not available (removed) |
| **Tasks API** | ✅ Available (required model) |
| **Migration necessity** | ✅ Yes, forced by package evolution |
| **Simplest approach** | ✅ Download model + use Tasks API |
| **Maintenance burden** | Low (no version downgrades needed) |

## What to Do Now

1. **Accept**: Tasks API is the only practical path forward
2. **Obtain**: Download `hand_landmarker.tflite` (80-90 MB)
   - Official source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
3. **Place**: `models/recognition/hand_landmarker.tflite`
4. **Run**: Preprocessing with `--model-path` argument

## Files to Update

The current implementation is already correct. Once you have the model:

```bash
# Current preprocessing script already supports this:
./.venv/bin/python scripts/preprocess_asl.py \
  --input-dir data/raw/asl_alphabet/asl_alphabet_train \
  --output-dir data/processed \
  --model-path models/recognition/hand_landmarker.tflite
```

No code changes needed — just need the model file!

## Conclusion

**The migration to Tasks API was necessary and correct.** There is no "simpler" approach using the legacy API because:
1. Legacy API doesn't exist in any available MediaPipe version
2. Tasks API is the official replacement
3. The codebase is already properly updated to use Tasks API

**The only blocker is obtaining the external model file**, which is a one-time download and setup task.
