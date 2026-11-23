---
license: apache-2.0
library_name: scikit-learn
tags:
  - color
  - color-naming
  - nearest-neighbors
  - CIELAB
---

# Stickdazzle Color Name

Traditional machine-learning model for predicting a descriptive color name from an sRGB hex value. It uses CIELAB nearest-neighbor retrieval and does not use generative AI.

Official website: [stickdazzle.com](https://www.stickdazzle.com/)

Author: Manojkumar Muthukumar ([GitHub](https://github.com/manoj10101996))

## Quick start

```bash
pip install -r requirements.txt
python predict.py "#7F4A36"
```

## Train

```bash
python train_color_name_model.py train color_names_training.csv color_name_model_extended.joblib --hex-col hex --name-col name --neighbors 5
```

`color_names_training.csv` is the single canonical training table. It combines the core and curated sources, plus ColorNames.org records with at least 5 votes so that one-vote community labels do not dominate nearest-neighbor retrieval.

The extended model is available as `color_name_model_extended.joblib`; the original `color_name_model.joblib` is retained as a legacy artifact.

## Details

- Input: `#RRGGBB` or `RRGGBB`
- Features: CIELAB calculated from sRGB
- Distance: Euclidean Delta E 1976
- Training rows: 208,960 unique hex/name pairs
- Artifact: `color_name_model_extended.joblib`

The result is an approximate label from the training data, not a universal name. Keep `NOTICE.md`, `source_audit.csv`, and `licenses/` when redistributing.

## Intended use and limitations

This model is intended for approximate color-name suggestions in applications, design tools, and exploratory analysis. It is not a colorimeter, a standards-compliance tool, or a substitute for official commercial color references.

Names come from heterogeneous public datasets and include community-created labels. Predictions can reflect duplicate names, subjective naming, cultural context, and source-specific biases. The model has not been evaluated against a human-labeled holdout set, so no accuracy or confidence claim is made. Treat the returned name and alternatives as suggestions and use the hex value as the authoritative color representation.

## Reproducibility

The released extended model was trained with `scikit-learn 1.8.0`, `numpy 2.3.5`, and `joblib 1.5.3` using `color_names_training.csv` and 5-neighbor CIELAB retrieval. Source provenance and licensing details are recorded in `NOTICE.md` and `source_audit.csv`.

## Attribution and licensing

The training and inference code and the released model are provided under the Apache License 2.0. The bundled datasets include material from multiple upstream sources with different licenses. See `NOTICE.md`, `source_audit.csv`, and `licenses/` for source attribution, license status, and reuse notes. Upstream data terms continue to apply to the corresponding data and derived artifacts.
