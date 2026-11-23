# Notices and attribution

## Project

Stickdazzle Color Name is maintained by Manojkumar Muthukumar. Official website: https://www.stickdazzle.com/. Author profile: https://github.com/manoj10101996.

The training and inference code and the released model are licensed under the Apache License 2.0; see `LICENSE`. This notice covers attribution and provenance for the bundled upstream data. It does not replace or change the license terms of those upstream sources.

## Core data

See `source_audit.csv` and `licenses/` for included source licenses and review status. Data from sources marked `review` is retained only as audit material and is not merged into the core training table.

## ColorNames.org community-voted snapshot

Source: https://colornames.org/download/colornames.zip (snapshot downloaded 2026-11-23). Project documentation says the downloadable data is CC0 1.0. See https://github.com/lgnbhl/colornames. This export contains submitted/voted color names and vote totals; names are community-created and may be informal.

## Coloripedia

Color data by Coloripedia: https://coloripedia.com. Dataset source: https://github.com/pacel1/coloripedia-data. Licensed under CC BY 4.0; attribution is required. Full license text is in `licenses/coloripedia_CC-BY-4.0_LICENSE.txt`.

## A Dictionary of Colour Combinations

Source: https://github.com/vtmade/dictionary-of-colour. The repository declares MIT; full license text is in `licenses/dictionary_of_colour_combinations_MIT_LICENSE.md`. The dataset adapts material from Sanzo Wada's published book and includes RGB/hex values converted from print colors, which may differ from the physical swatches.

## Extended training package

`color_names_training.csv` is the single canonical training table. It includes all core and curated records, while ColorNames.org-only records are included when their snapshot vote count is at least 5. The resulting `color_name_model_extended.joblib` is a separate artifact; the original model is retained as a legacy artifact. The original source snapshots are not duplicated in the package; their provenance remains recorded in `source_audit.csv`.

## Additional provenance

The core training table also includes records from the included CC0, MIT, and other reviewed sources listed in `source_audit.csv`. The repository retains source-specific license texts in `licenses/` where available. Sources marked `review` or otherwise not confirmed for redistribution are retained as audit material or separate data and should not be treated as cleared for unrestricted reuse. Review `source_audit.csv` before redistributing the datasets or any derived model.
