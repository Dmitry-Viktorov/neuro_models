# Run Summary

## Configuration
- img: `C:\CODES\C++\foton\neuro_models\dataset\img`
- mask: `C:\CODES\C++\foton\neuro_models\dataset\mask`
- texture: `C:\CODES\C++\foton\neuro_models\dataset\texture`
- rgb_image: `None`
- tile: `256`
- stride: `128`
- infer_stride: `64`
- batch: `2`
- epochs: `12`
- learning_rate: `0.0001`
- patience: `3`
- seed: `42`

## Best Model
- model: `SegNetLite`
- test_f1: `0.853633`
- full_macro_f1: `0.843951`
- threshold: `0.550000`

## Ranking (by test_f1)
1. `SegNetLite`: test_f1=0.853633, full_macro_f1=0.843951, epochs=12
2. `YOLOSegLite`: test_f1=0.847724, full_macro_f1=0.842344, epochs=12
3. `ResUNet`: test_f1=0.842744, full_macro_f1=0.844139, epochs=9
4. `UNet`: test_f1=0.821725, full_macro_f1=0.815834, epochs=7

See also: `model_comparison.md` and per-model `metrics.txt` files.
