# Hybrid Crop Disease Detection Pipeline

This module extends the existing CSV-based crop recommendation system into a real-time image diagnosis workflow.

## Model Strategy

- The legacy production path is a single-head `MobileNetV2` classifier.
- The upgraded training path uses a joint crop+disease classifier with a primary pair-class head and an auxiliary crop head.
- The default upgraded backbone is `DenseNet121`, with `ResNet50V2` and `Xception` available for benchmarking.
- This joint design is more stable than the earlier experimental EfficientNet multitask script because it preserves exact crop+disease supervision instead of splitting crop and disease into loosely aligned label spaces.

## Folder-Style Dataset Assumption

The image pipeline expects PlantVillage-style folders:

```text
PlantVillage/
  Tomato___Early_blight/
  Tomato___healthy/
  Potato___Late_blight/
  Rice___Brown_spot/
```

You can pass more than one `--dataset-dir` when adding real-world images. Keep the same class-folder naming scheme so both sources merge cleanly.

## Training

Legacy single-head training:

```bash
python scripts/train_disease_model.py --dataset-dir data/plantvillage --dataset-dir data/real_world_leaf_images --backbone mobilenetv2 --image-size 224 --batch-size 32
```

Upgraded joint crop+disease training:

```bash
python scripts/train_multitask_local.py --dataset-dir "data/external/plantvillage dataset/color" --backbone densenet121 --image-size 224 --batch-size 16
```

Training pipeline features:

- 70/15/15 stratified split
- augmentation for brightness, rotation, zoom, blur, and Gaussian noise
- class weighting for imbalance
- warm-up training with frozen ImageNet backbone
- fine-tuning of deeper convolution blocks
- early stopping and learning-rate reduction

Legacy artifacts are written to:

- `models/disease/leaf_disease_classifier.keras`
- `models/disease/leaf_disease_metadata.json`
- `reports/disease/evaluation_metrics.json`
- `reports/disease/confusion_matrix.png`

Joint-model artifacts are written to:

- `models/disease/leaf_disease_joint_classifier.keras`
- `models/disease/leaf_disease_joint_metadata.json`
- `reports/disease_joint/evaluation_metrics.json`
- `reports/disease_joint/confusion_matrix.png`

## Hybrid Inference Logic

1. The vision model predicts probabilities for every `crop + disease` class.
2. The existing CSV model predicts crop priors from environmental features.
3. The hybrid predictor reweights disease probabilities using the crop prior.
4. Final output includes:

- crop name
- disease name
- confidence
- fertilizer hint from the tabular model
- basic treatment recommendation

This is implemented in `src/hybrid_disease_predictor.py`.

## Real-Time Camera Inference

Create a JSON file such as `crop_features.json`:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "pH": 6.5,
  "temperature": 26.0,
  "humidity": 82.0,
  "rainfall": 210.0,
  "soil_type": "Alluvial",
  "season": "Summer",
  "region": "Konkan"
}
```

Then run:

```bash
python scripts/run_realtime_detection.py --features-json crop_features.json
```

Use `--provided-crop Tomato` when the crop is already known and you want hard filtering.

## Explainability

Generate a Grad-CAM overlay:

```bash
python scripts/generate_gradcam.py --image-path sample_leaf.jpg
```

In the real-time app, press `G` to save a Grad-CAM snapshot of the current frame.

## TensorFlow Lite Conversion

```bash
python scripts/convert_to_tflite.py --quantization float16
```

This creates `models/disease/leaf_disease_classifier.tflite`, which is the recommended mobile artifact for deployment.

## Suggestions For Indian Crop Expansion

- add rice, wheat, maize, cotton, sugarcane, chili, and pomegranate field images from Indian conditions
- collect smartphone photos under harsh sunlight, shadows, and mixed backgrounds
- include district or agro-climatic zone metadata for Maharashtra and similar regions
- add multilingual remedies in English, Hindi, and Marathi for farmer-facing deployment

## Maharashtra Coverage Planning

The repository now includes `data/maharashtra_disease_targets.json`, which lists the current target crop set for Maharashtra-focused disease expansion. Use it as the canonical planning file for:

- deciding which crops must be supported before demo or production claims
- naming new training folders in `Crop___Disease_Name` format
- checking whether the current trained metadata actually covers the target list

Run the audit script:

```bash
python scripts/check_disease_coverage.py
```

This script compares the trained disease metadata against the Maharashtra target list and reports which crops and classes are still missing.
