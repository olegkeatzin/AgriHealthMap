"""
Test script to verify all three models are properly integrated
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all model modules can be imported"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from crop_model import CropDetectionModel
        print("[OK] crop_model imported successfully")
    except Exception as e:
        print(f"[FAIL] crop_model import failed: {e}")
        return False

    try:
        from field_segmentation_model import FieldSegmentationModel
        print("[OK] field_segmentation_model imported successfully")
    except Exception as e:
        print(f"[FAIL] field_segmentation_model import failed: {e}")
        return False

    try:
        from ndvi_prediction_model import NDVIPredictionModel
        print("[OK] ndvi_prediction_model imported successfully")
    except Exception as e:
        print(f"[FAIL] ndvi_prediction_model import failed: {e}")
        return False

    return True


def test_model_initialization():
    """Test that all models can be initialized"""
    print("\n" + "=" * 60)
    print("Testing model initialization...")
    print("=" * 60)

    from crop_model import CropDetectionModel
    from field_segmentation_model import FieldSegmentationModel
    from ndvi_prediction_model import NDVIPredictionModel

    models_ok = True

    # Test Crop Detection Model
    try:
        crop_model = CropDetectionModel(
            model_path='models/crop_detection_model.pt',
            device='cpu'
        )
        print("[OK] Crop Detection Model initialized")
        print(f"     - Classes: {len(crop_model.CLASS_NAMES)}")
        print(f"     - Device: {crop_model.device}")
    except Exception as e:
        print(f"[FAIL] Crop Detection Model failed: {e}")
        models_ok = False

    # Test Field Segmentation Model
    try:
        field_model = FieldSegmentationModel(
            model_path='models/field_segmentation_model.pth',
            device='cpu'
        )
        print("[OK] Field Segmentation Model initialized")
        print(f"     - Classes: {len(field_model.CLASS_NAMES)}")
        print(f"     - Device: {field_model.device}")
    except Exception as e:
        print(f"[FAIL] Field Segmentation Model failed: {e}")
        models_ok = False

    # Test NDVI Prediction Model
    try:
        ndvi_model = NDVIPredictionModel(
            model_path='models/ndvi_lstm_best.pth',
            device='cpu',
            model_type='lstm'
        )
        print("[OK] NDVI Prediction Model initialized")
        print(f"     - Model type: {ndvi_model.model_type}")
        print(f"     - Device: {ndvi_model.device}")
        print(f"     - Sequence length: {ndvi_model.sequence_length}")
        print(f"     - Forecast horizon: {ndvi_model.forecast_horizon}")
    except Exception as e:
        print(f"[FAIL] NDVI Prediction Model failed: {e}")
        models_ok = False

    return models_ok


def test_model_files():
    """Test that all model weight files exist"""
    print("\n" + "=" * 60)
    print("Testing model files...")
    print("=" * 60)

    files_ok = True

    model_files = [
        'models/crop_detection_model.pt',
        'models/field_segmentation_model.pth',
        'models/ndvi_lstm_best.pth'
    ]

    for filepath in model_files:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[OK] {filepath} exists ({size_mb:.2f} MB)")
        else:
            print(f"[FAIL] {filepath} not found")
            files_ok = False

    return files_ok


def test_config_files():
    """Test that all config files exist"""
    print("\n" + "=" * 60)
    print("Testing config files...")
    print("=" * 60)

    configs_ok = True

    config_files = [
        'configs/crop_config.json',
        'configs/field_config.json',
        '.env'
    ]

    for filepath in config_files:
        if os.path.exists(filepath):
            print(f"[OK] {filepath} exists")
        else:
            print(f"[FAIL] {filepath} not found")
            configs_ok = False

    return configs_ok


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AgriHealthMap - Model Integration Test")
    print("=" * 60)

    results = {
        'imports': False,
        'files': False,
        'configs': False,
        'initialization': False
    }

    # Run tests
    results['files'] = test_model_files()
    results['configs'] = test_config_files()
    results['imports'] = test_imports()

    if results['imports']:
        results['initialization'] = test_model_initialization()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name.capitalize()}: [{status}]")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All tests passed!")
    else:
        print("FAILURE: Some tests failed. Check output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
