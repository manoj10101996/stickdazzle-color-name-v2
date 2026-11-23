# Color-name training data sources

Core dataset: **34,080 source-deduplicated hex/name rows** from the previously included sources. It is intentionally kept apart from optional sources with different naming styles or label semantics.

## Canonical training data

`color_names_training.csv` is the single canonical training table. It contains 208,960 unique `hex,name` pairs from the core data, Coloripedia, *A Dictionary of Colour Combinations*, and ColorNames.org records with at least 5 votes. `source_audit.csv` records source URLs, licenses, counts, and decisions.

## Core files

- `color_names_training.csv`: the only canonical model-training CSV.
- `source_audit.csv`: source links, extraction counts, and inclusion/license decisions.
- `color_name_model_extended.joblib`: model trained from `color_names_training.csv`; the original `color_name_model.joblib` is unchanged.
- `NOTICE.md` and `licenses/`: attribution and license notices for included sources.

The original source snapshots are no longer duplicated in the package; their provenance and licensing details remain in `source_audit.csv` and `NOTICE.md`. Several supplied links point to palette collections or names without hex values; those cannot directly train the `hex -> color name` model. The Riso data remains excluded due to its stated non-commercial terms.
