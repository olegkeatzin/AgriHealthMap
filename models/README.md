# Model Weights

This directory contains the trained model weights for AgriHealthMap.

## Models

The following models are required for the application to work:

1. **crop_detection_model.pt** (477 MB) - TorchScript model for crop detection
2. **field_segmentation_model.pth** (1440 MB) - PyTorch model for field segmentation
3. **ndvi_lstm_best.pth** (0.2 MB) - PyTorch LSTM model for NDVI prediction

**Total size:** ~1.9 GB

## Why are models not in the repository?

The model files are too large to be stored directly in GitHub (which has a 100 MB file size limit). Therefore, they are excluded via `.gitignore`.

## How to obtain the models

### Option 1: Download from source
If you trained these models yourself, copy them from your training directory:

```bash
# From the project root
cp D:/Hakaton/crop_detection_model.pt models/
cp D:/Hakaton/last_checkpoint.pth models/field_segmentation_model.pth
cp D:/Hakaton/models/ndvi_lstm_best.pth models/
```

### Option 2: Download from cloud storage
If the models are hosted on cloud storage (Google Drive, Dropbox, etc.), download them and place them in this directory.

### Option 3: Use Git LFS (if configured)
If Git LFS is set up for this repository, the models will be downloaded automatically when you clone the repo.

## Verification

After placing the models in this directory, verify they are loaded correctly:

```bash
cd D:\Hakaton\Argo_Health\AgriHealthMap
python test_models.py
```

You should see:
```
SUCCESS: All tests passed!
```

## Model Details

### Crop Detection Model
- **File:** crop_detection_model.pt
- **Format:** TorchScript
- **Size:** 477 MB
- **Architecture:** Attention U-Net (5 levels)
- **Classes:** 6 (background, wheat, corn, sunflower, soybean, other_crops)

### Field Segmentation Model
- **File:** field_segmentation_model.pth
- **Format:** PyTorch checkpoint
- **Size:** 1440 MB
- **Architecture:** Attention U-Net with Attention Gates (5 levels)
- **Classes:** 5 (background, field, field_boundary, other, undefined)

### NDVI Prediction Model
- **File:** ndvi_lstm_best.pth
- **Format:** PyTorch checkpoint
- **Size:** 0.2 MB
- **Architecture:** LSTM (2 layers) + FC layers
- **Input:** Time series (sequence_length=5, features=2)
- **Output:** Prediction 2 steps ahead

## Configuration

Model paths are configured in the `.env` file:

```env
CROP_MODEL_PATH=models/crop_detection_model.pt
FIELD_MODEL_PATH=models/field_segmentation_model.pth
NDVI_MODEL_PATH=models/ndvi_lstm_best.pth
```

## Troubleshooting

### Models not loading
1. Check that all three model files are in this directory
2. Run `python test_models.py` to verify
3. Check that file names match exactly

### File not found errors
Make sure the `.env` file has the correct paths relative to the project root.

### CUDA out of memory
If you're using GPU and run out of memory, switch to CPU in `main.py`:
```python
crop_model = CropDetectionModel(..., device='cpu')
field_model = FieldSegmentationModel(..., device='cpu')
ndvi_model = NDVIPredictionModel(..., device='cpu')
```

## Training

To retrain these models, refer to the training scripts in:
- `D:\Hakaton\scripts\training\train_ndvi_prediction.py`
- Other training scripts in the original project directory

## Support

For issues or questions, see:
- `INTEGRATION_COMPLETE.md` - Full integration documentation
- `QUICKSTART.md` - Quick start guide
- `COMPLETE_API_GUIDE.md` - API documentation
