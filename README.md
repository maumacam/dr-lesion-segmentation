# Diabetic Retinopathy Lesion Segmentation — Project Scaffold (DIARETDB1 Only)

This updated scaffold assumes we only have the **DIARETDB1 - Standard Diabetic Retinopathy Database**. The pipeline, models, and web app remain the same, but data preparation steps are adjusted for this dataset.

---

## Repository root (updated)

```
dr-lesion-segmentation/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── data/
│   ├── diaretdb1/            # raw images and coordinate annotations from Kaggle dataset
│   └── processed/            # generated masks and preprocessed images
├── training/
│   ├── __init__.py
│   ├── data_loader.py        # loads DIARETDB1 and generates lesion masks
│   ├── augmentations.py      # albumentations recipes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segnet.py
│   │   └── unetpp.py
│   ├── losses.py             # BCE + soft-dice implementation
│   ├── train.py              # training loop
│   └── utils.py              # helpers
|   |__preprocess.py
├── evaluation/
│   ├── evaluate.py
│   ├── stats_test.py
│   └── visualize.py
├── inference/
│   ├── predict.py
│   └── export_model.py
├── webapp/
│   ├── backend/
│   │   ├── app/main.py
│   │   ├── app/model_server.py
│   │   └── app/schemas.py
│   │   └── Dockerfile
│   └── frontend/
│       ├── package.json
│       ├── public/
│       └── src/
│           ├── App.jsx
│           ├── components/
│           └── styles/
└── thesis/
    ├── figures/
    └── appendix/
```

---

## README (updated summary)

### Project overview

A reproducible pipeline for lesion-level segmentation of diabetic retinopathy using SegNet and U-Net++ implemented in TensorFlow/Keras, trained and evaluated **only on DIARETDB1** dataset. Models predict 4 binary masks (MA, HE, SE, EX).

### Data preparation (DIARETDB1)

* Place Kaggle dataset under `data/diaretdb1/`.
* Run `training/data_loader.py --prepare` to:

  * Load raw images and lesion coordinates.
  * Convert coordinates to pixel masks (4 channels: MA, HE, SE, EX).
  * Resize images/masks to 512×512 and normalize.
  * Save processed files in `data/processed/`.
  * Generate QA overlays for visual verification.

**Masking strategy:**

* Microaneurysm (MA): small disk radius (\~2–3 px @ original res).
* Hemorrhages (HE), Soft Exudates (SE), Hard Exudates (EX): larger radius (\~5–15 px) or polygon fill if shape data present.

### Training & Evaluation

* Train SegNet or U-Net++ using:

```bash
python training/train.py --model segnet --epochs 100 --batch-size 8
```

* Evaluate metrics per lesion type (Dice, IoU, Sens, Spec) via `evaluation/evaluate.py`.

### Inference & Web App

* FastAPI backend for inference.
* React frontend for image upload, overlays, and per-class metrics in a light theme.

### Limitations

* Single dataset → less generalization. Note this in the thesis discussion.

---

## Next Steps

1. Implement `training/data_loader.py`:

   * Load DIARETDB1 structure from Kaggle.
   * Parse coordinate files.
   * Convert coordinates to masks.
   * Save QA overlays for verification.
2. Implement SegNet and U-Net++ models.
3. Training script (`train.py`).
4. Backend and frontend integration.

---

I will now create the full **`training/data_loader.py`** code for DIARETDB1, including:

* Reading Kaggle structure.
* Coordinate-to-mask logic.
* Visual QA overlay generation.
* CLI for preprocessing and saving data.
