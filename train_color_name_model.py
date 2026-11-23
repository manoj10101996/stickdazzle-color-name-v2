#!/usr/bin/env python3
"""Train and query a nearest-neighbor color-name model.

Supported training data:
  CSV: columns such as `hex,name` or `color,color_name`
  JSON object: {"#rrggbb": "Color Name", ...}
  JSON list: [{"hex": "#rrggbb", "name": "Color Name"}, ...]

The model uses CIELAB (D65) coordinates and Euclidean Delta E 1976 distance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

HEX_PATTERN = re.compile(r"^#?([0-9a-fA-F]{6})$")
HEX_COLUMN_CANDIDATES = ("hex", "hex_code", "hex_color", "hex_value", "color_hex", "colour_hex", "color", "colour", "code", "value")
NAME_COLUMN_CANDIDATES = ("name", "color_name", "colour_name", "label", "colorname", "colourname")
MODEL_FORMAT_VERSION = 1


def normalize_hex(value: Any) -> str:
    match = HEX_PATTERN.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid hex color {value!r}; expected #RRGGBB")
    return "#" + match.group(1).lower()


def hex_to_rgb(hex_value: str) -> np.ndarray:
    value = normalize_hex(hex_value)[1:]
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] values to CIELAB using the D65 white point."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ rgb_to_xyz.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    x, y, z = np.moveaxis(f, -1, 0)
    return np.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), axis=-1)


def _record(hex_value: Any, name: Any, location: str) -> tuple[str, str]:
    color = normalize_hex(hex_value)
    label = str(name).strip()
    if not label:
        raise ValueError(f"Empty color name at {location}")
    return color, label


def _find_column(headers: list[str], requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in headers:
            raise ValueError(f"Column {requested!r} not found. Available columns: {headers}")
        return requested
    by_normalized = {h.strip().lower().replace(" ", "_"): h for h in headers}
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]
    raise ValueError(
        f"Could not identify the {kind} column in {headers}. "
        f"Pass it explicitly with --{kind}-col."
    )


def load_records(path: str | Path, hex_col: str | None = None, name_col: str | None = None) -> list[tuple[str, str]]:
    """Read and validate the supported JSON/CSV training formats."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        records: list[tuple[str, str]] = []
        if isinstance(content, dict):
            for i, (color, name) in enumerate(content.items(), start=1):
                records.append(_record(color, name, f"JSON item {i}"))
        elif isinstance(content, list):
            for i, row in enumerate(content, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"JSON item {i} must be an object")
                headers = list(row)
                color_key = _find_column(headers, hex_col, HEX_COLUMN_CANDIDATES, "hex")
                label_key = _find_column(headers, name_col, NAME_COLUMN_CANDIDATES, "name")
                records.append(_record(row.get(color_key), row.get(label_key), f"JSON item {i}"))
        else:
            raise ValueError("JSON training data must be an object or a list of objects")
        return records

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file needs a header row")
        color_key = _find_column(reader.fieldnames, hex_col, HEX_COLUMN_CANDIDATES, "hex")
        label_key = _find_column(reader.fieldnames, name_col, NAME_COLUMN_CANDIDATES, "name")
        records = []
        for line_number, row in enumerate(reader, start=2):
            records.append(_record(row.get(color_key), row.get(label_key), f"CSV line {line_number}"))
        return records


def train(data_path: str, model_path: str, hex_col: str | None = None, name_col: str | None = None, neighbors: int = 5) -> dict:
    if neighbors < 1:
        raise ValueError("--neighbors must be at least 1")
    rows = load_records(data_path, hex_col=hex_col, name_col=name_col)
    source_row_count = len(rows)
    if not rows:
        raise ValueError("No training records were found")

    # Collapse repeated copies of the same pair so one source cannot dominate
    # the neighbor vote just because it duplicated records from another source.
    rows = sorted(set(rows), key=lambda row: (row[0], row[1].casefold(), row[1]))
    hexes = [row[0] for row in rows]
    names = [row[1] for row in rows]
    rgbs = np.stack([hex_to_rgb(value) for value in hexes])
    labs = rgb_to_lab(rgbs)

    index = NearestNeighbors(n_neighbors=min(neighbors, len(rows)), algorithm="brute", metric="euclidean")
    index.fit(labs)
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "index": index,
        "hexes": hexes,
        "names": names,
        "neighbors": min(neighbors, len(rows)),
        "metadata": {
            "model_type": "CIELAB k-nearest-neighbor color-name predictor",
            "feature_space": "CIELAB D65",
            "distance": "Euclidean Delta E 1976",
            "source_rows": source_row_count,
            "unique_hex_name_pairs": len(rows),
        },
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    return bundle["metadata"]


def predict(hex_value: str, model_path: str, top: int = 3) -> dict:
    if top < 1:
        raise ValueError("--top must be at least 1")
    bundle = joblib.load(model_path)
    if bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported or outdated model file; retrain it with this script")

    color = normalize_hex(hex_value)
    lab = rgb_to_lab(hex_to_rgb(color).reshape(1, 3))
    distances, indices = bundle["index"].kneighbors(lab, n_neighbors=bundle["neighbors"])
    distances, indices = distances[0], indices[0]

    # Exact hex matches take precedence. If sources assigned multiple names to
    # that exact hex, the alphabetically first name wins the deterministic tie.
    exact = [int(i) for d, i in zip(distances, indices) if d < 1e-9]
    if exact:
        exact_names = sorted({bundle["names"][i] for i in exact}, key=lambda s: (s.casefold(), s))
        chosen = exact_names[0]
        return {
            "input_hex": color,
            "predicted_name": chosen,
            "exact_match": True,
            "nearest_known_hex": color,
            "delta_e_76": 0.0,
            "alternatives": [{"name": n, "neighbor_vote_score": round(1 / len(exact_names), 4)} for n in exact_names[:top]],
        }

    # Inverse-distance voting lets several nearby records support a name.
    scores: dict[str, float] = defaultdict(float)
    nearest_for_name: dict[str, tuple[float, int]] = {}
    for distance, i in zip(distances, indices):
        i = int(i)
        name = bundle["names"][i]
        scores[name] += 1.0 / max(float(distance), 1e-6)
        if name not in nearest_for_name or distance < nearest_for_name[name][0]:
            nearest_for_name[name] = (float(distance), i)

    ranked = sorted(scores, key=lambda n: (-scores[n], n.casefold(), n))
    total_score = sum(scores.values())
    alternatives = []
    for name in ranked[:top]:
        distance, i = nearest_for_name[name]
        alternatives.append({
            "name": name,
            "nearest_known_hex": bundle["hexes"][i],
            "delta_e_76": round(distance, 3),
            # This is a normalized neighbor vote, not a calibrated probability.
            "neighbor_vote_score": round(scores[name] / total_score, 4),
        })
    best = alternatives[0]
    return {
        "input_hex": color,
        "predicted_name": best["name"],
        "exact_match": False,
        "nearest_known_hex": best["nearest_known_hex"],
        "delta_e_76": best["delta_e_76"],
        "alternatives": alternatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and use a hex-to-color-name ML model")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train from CSV or JSON name/hex data")
    train_parser.add_argument("data", help="training CSV or JSON file")
    train_parser.add_argument("model", help="output .joblib model path")
    train_parser.add_argument("--hex-col", help="hex column name for CSV/JSON rows")
    train_parser.add_argument("--name-col", help="name column name for CSV/JSON rows")
    train_parser.add_argument("--neighbors", type=int, default=5, help="number of nearest records to vote (default: 5)")

    predict_parser = commands.add_parser("predict", help="predict the name of one hex color")
    predict_parser.add_argument("hex", help="color such as #3A7BD5")
    predict_parser.add_argument("model", help="trained .joblib model path")
    predict_parser.add_argument("--top", type=int, default=3, help="number of alternative names to return")

    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.data, args.model, args.hex_col, args.name_col, args.neighbors)
        else:
            result = predict(args.hex, args.model, args.top)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
