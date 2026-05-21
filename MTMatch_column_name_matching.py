import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import torch
from transformers import AutoTokenizer

from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR, lm_mp


MISSING_MARKERS = {"", "NA", "NaN", "nan", "null", "None"}


def normalize_column_name(name: str) -> str:
    if name is None:
        return ""
    return str(name).replace("\ufeff", "").strip()


def read_table_columns(
    csv_path: str,
    max_rows: int = 50000,
) -> Dict[str, List[str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV file: {csv_path}")

        normalized_fieldnames = [normalize_column_name(column_name) for column_name in reader.fieldnames]
        reader.fieldnames = normalized_fieldnames

        column_values: Dict[str, List[str]] = {
            column_name: [] for column_name in normalized_fieldnames
        }

        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break
            for column_name, value in row.items():
                normalized_name = normalize_column_name(column_name)
                if value is None:
                    column_values[normalized_name].append("")
                else:
                    column_values[normalized_name].append(str(value))

    return column_values


def detect_column_type(values: List[str], max_samples: int = 2000) -> str:
    non_empty_values: List[str] = [
        value
        for value in values
        if value is not None and value.strip() not in MISSING_MARKERS
    ]
    if not non_empty_values:
        return "empty"

    sample_values = non_empty_values[:max_samples]
    numeric_count = 0
    date_like_count = 0

    for value_text in sample_values:
        stripped = value_text.strip()
        try:
            float(stripped.replace(",", ""))
            numeric_count += 1
            continue
        except ValueError:
            pass

        has_digit = any(character.isdigit() for character in stripped)
        has_separator = any(sep in stripped for sep in ("/", "-", ":"))
        if has_digit and has_separator:
            date_like_count += 1

    total = len(sample_values)
    numeric_ratio = numeric_count / total
    date_ratio = date_like_count / total

    if numeric_ratio >= 0.8:
        return "numeric"
    if date_ratio >= 0.8:
        return "date_or_time"

    unique_values = set(sample_values)
    if len(unique_values) <= 0.5 * total:
        return "categorical"
    return "text"


def compute_numeric_features(values: List[str], max_samples: int = 50000) -> List[float]:
    numeric_values: List[float] = []
    missing_count = 0

    for value_index, value_text in enumerate(values):
        if max_samples is not None and value_index >= max_samples:
            break
        if value_text is None:
            missing_count += 1
            continue
        stripped = value_text.strip()
        if stripped in MISSING_MARKERS:
            missing_count += 1
            continue
        try:
            numeric_value = float(stripped.replace(",", ""))
            numeric_values.append(numeric_value)
        except ValueError:
            missing_count += 1

    if not numeric_values:
        return [0.0] * 8

    array = np.asarray(numeric_values, dtype=np.float64)
    mean_value = float(np.mean(array))
    std_value = float(np.std(array))
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    q25, q50, q75 = np.percentile(array, [25, 50, 75])

    total_count = len(values)
    missing_ratio = missing_count / total_count if total_count > 0 else 0.0

    return [
        mean_value,
        std_value,
        min_value,
        max_value,
        float(q25),
        float(q50),
        float(q75),
        missing_ratio,
    ]


def compute_categorical_values(
    values: List[str],
    max_samples: int = 50000,
    max_unique: int = 5000,
) -> Set[str]:
    tokens: List[str] = []
    for value_index, value_text in enumerate(values):
        if max_samples is not None and value_index >= max_samples:
            break
        if value_text is None:
            continue
        stripped = value_text.strip()
        if not stripped or stripped in MISSING_MARKERS:
            continue
        tokens.append(stripped.lower())

    if not tokens:
        return set()

    value_counter = Counter(tokens)
    most_common_values = value_counter.most_common(max_unique)
    return {item for item, _ in most_common_values}


def numeric_similarity(
    features_left: List[float],
    features_right: List[float],
) -> float:
    if not features_left or not features_right:
        return 0.0
    if len(features_left) != len(features_right):
        return 0.0

    normalized_differences: List[float] = []
    for feature_left, feature_right in zip(features_left, features_right):
        denominator = abs(feature_left) + abs(feature_right) + 1e-8
        normalized_differences.append(abs(feature_left - feature_right) / denominator)

    average_distance = float(sum(normalized_differences) / len(normalized_differences))
    return 1.0 / (1.0 + average_distance)


def categorical_similarity(
    values_left: Set[str],
    values_right: Set[str],
) -> float:
    if not values_left or not values_right:
        return 0.0
    intersection_size = len(values_left & values_right)
    union_size = len(values_left | values_right)
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def build_column_profiles(
    csv_path: str,
    max_rows: int = 50000,
) -> Dict[str, Dict[str, Any]]:
    raw_columns = read_table_columns(csv_path, max_rows=max_rows)
    profiles: Dict[str, Dict[str, Any]] = {}

    for column_name, column_values in raw_columns.items():
        column_type = detect_column_type(column_values)
        profile: Dict[str, Any] = {"type": column_type}
        if column_type == "numeric":
            profile["numeric_features"] = compute_numeric_features(column_values)
            profile["categorical_values"] = set()
        else:
            profile["numeric_features"] = []
            profile["categorical_values"] = compute_categorical_values(column_values)
        profiles[column_name] = profile

    return profiles


def column_similarity(
    profile_left: Dict[str, Any],
    profile_right: Dict[str, Any],
) -> float:
    type_left = profile_left.get("type")
    type_right = profile_right.get("type")

    if type_left == "numeric" and type_right == "numeric":
        return numeric_similarity(
            profile_left.get("numeric_features", []),
            profile_right.get("numeric_features", []),
        )

    non_numeric_types = {"categorical", "text", "date_or_time"}
    if type_left in non_numeric_types and type_right in non_numeric_types:
        return categorical_similarity(
            profile_left.get("categorical_values", set()),
            profile_right.get("categorical_values", set()),
        )

    return 0.0


def match_columns_heuristic(
    profiles_reference: Dict[str, Dict[str, Any]],
    profiles_target: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    matches: List[Tuple[str, str, float]] = []
    for target_name, target_profile in profiles_target.items():
        best_reference_name = None
        best_similarity = -math.inf
        for reference_name, reference_profile in profiles_reference.items():
            similarity_score = column_similarity(reference_profile, target_profile)
            if similarity_score > best_similarity:
                best_similarity = similarity_score
                best_reference_name = reference_name
        if best_reference_name is None:
            best_reference_name = ""
            best_similarity = 0.0
        matches.append((target_name, best_reference_name, best_similarity))
    matches.sort(key=lambda item: item[2], reverse=True)
    return matches


def build_column_texts(
    csv_path: str,
    max_rows: int,
    max_cells: int,
    include_table_name: bool = False,
    include_column_name: bool = True,
) -> Tuple[List[str], List[str]]:
    raw_columns = read_table_columns(csv_path, max_rows=max_rows)
    column_names: List[str] = list(raw_columns.keys())
    table_name = Path(csv_path).name
    texts: List[str] = []
    for name in column_names:
        values = raw_columns[name]
        tokens: List[str] = []
        for value_text in values:
            if value_text is None:
                continue
            stripped = value_text.strip()
            if not stripped or stripped in MISSING_MARKERS:
                continue
            tokens.append(stripped)
            if len(tokens) >= max_cells:
                break

        if include_table_name or include_column_name:
            text_parts: List[str] = []
            if include_table_name:
                text_parts.extend(["table", table_name])
            if include_column_name:
                text_parts.extend(["column", name])
            if tokens:
                text_parts.extend(["values", " ".join(tokens)])
            texts.append(" ".join(text_parts).strip())
        else:
            texts.append(" ".join(tokens))
    return column_names, texts


class _BTConfig:
    def __init__(self, lm: str, projector: str, batch_size: int, lm_only: bool) -> None:
        self.lm = lm
        self.projector = projector
        self.batch_size = batch_size
        self.lm_only = lm_only
        self.task_type = "em"
        self.scale_loss = 1.0 / 256.0
        self.lambd = 3.9e-3
        self.alpha_bt = 0.001


def encode_with_bt(
    texts: List[str],
    lm: str,
    ckpt_path: str,
    batch_size: int,
    max_len: int,
    lm_only: bool,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = _BTConfig(lm=lm, projector="768", batch_size=batch_size, lm_only=lm_only)

    model = BarlowTwinsSimCLR(hp, device=device, lm=lm)
    if ckpt_path and not lm_only:
        saved_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        #model.load_state_dict(saved_state["model"],strict=False)
        state_dict = saved_state["model"]
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
        
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])

    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            outputs = model.bert(input_ids)[0][:, 0, :]
            all_embeddings.append(outputs.cpu().numpy())

    if not all_embeddings:
        return np.zeros((0, 768), dtype=np.float32)

    mat = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat = mat / norms
    return mat


def match_columns_bt(
    reference_csv: str,
    target_csv: str,
    max_rows: int,
    max_cells: int,
    lm: str,
    ckpt_path: str,
    batch_size: int,
    max_len: int,
    lm_only: bool,
) -> List[Tuple[str, str, float]]:
    ref_names, ref_texts = build_column_texts(reference_csv, max_rows, max_cells)
    tgt_names, tgt_texts = build_column_texts(target_csv, max_rows, max_cells)

    ref_embeddings = encode_with_bt(ref_texts, lm, ckpt_path, batch_size, max_len, lm_only)
    tgt_embeddings = encode_with_bt(tgt_texts, lm, ckpt_path, batch_size, max_len, lm_only)

    if not len(ref_embeddings) or not len(tgt_embeddings):
        return []

    similarity_matrix = np.matmul(tgt_embeddings, ref_embeddings.T)

    matches: List[Tuple[str, str, float]] = []
    for tgt_index, tgt_name in enumerate(tgt_names):
        row = similarity_matrix[tgt_index]
        best_ref_index = int(np.argmax(row))
        best_score = float(row[best_ref_index])
        best_ref_name = ref_names[best_ref_index]
        matches.append((tgt_name, best_ref_name, best_score))

    matches.sort(key=lambda item: item[2], reverse=True)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match table column names based on column data characteristics. "
            "Can use a heuristic stats-based backend or the MTMatch self-supervised encoder."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference CSV file whose column names are treated as canonical.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target CSV file whose column names need to be matched.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=50000,
        help="Maximum number of rows from each table to use when computing statistics.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="heuristic",
        choices=["heuristic", "bt"],
        help="Matching backend: heuristic stats or BarlowTwins/SimCLR encoder.",
    )
    parser.add_argument(
        "--bt_lm",
        type=str,
        default="roberta",
        help="Backbone LM name for the MTMatch encoder.",
    )
    parser.add_argument(
        "--bt_ckpt_path",
        type=str,
        default="",
        help="Optional path to a trained ssl.pt checkpoint. If empty, use LM-only.",
    )
    parser.add_argument(
        "--bt_batch_size",
        type=int,
        default=32,
        help="Batch size when encoding columns with the MTMatch encoder.",
    )
    parser.add_argument(
        "--bt_max_len",
        type=int,
        default=128,
        help="Max sequence length for MTMatch encoder tokenization.",
    )
    parser.add_argument(
        "--max_cells_per_column",
        type=int,
        default=128,
        help="Maximum number of non-empty cells per column when building serialized texts.",
    )
    parser.add_argument(
        "--bt_lm_only",
        action="store_true",
        help="If set, ignore ssl checkpoint and use LM-only encoder.",
    )

    args = parser.parse_args()

    if args.backend == "heuristic":
        print(f"Building column profiles for reference table: {args.reference}")
        reference_profiles = build_column_profiles(args.reference, max_rows=args.max_rows)
        print(f"Found {len(reference_profiles)} columns in reference table.")

        print(f"Building column profiles for target table: {args.target}")
        target_profiles = build_column_profiles(args.target, max_rows=args.max_rows)
        print(f"Found {len(target_profiles)} columns in target table.")

        print("Matching columns based on heuristic statistics...")
        matches = match_columns_heuristic(reference_profiles, target_profiles)
    else:
        lm_only = args.bt_lm_only or not args.bt_ckpt_path
        print("Matching columns using the MTMatch BarlowTwins/SimCLR encoder...")
        matches = match_columns_bt(
            reference_csv=args.reference,
            target_csv=args.target,
            max_rows=args.max_rows,
            max_cells=args.max_cells_per_column,
            lm=args.bt_lm,
            ckpt_path=args.bt_ckpt_path,
            batch_size=args.bt_batch_size,
            max_len=args.bt_max_len,
            lm_only=lm_only,
        )

    print("\nMatched columns (target -> reference, similarity):")
    for target_name, reference_name, similarity_score in matches:
        print(
            f"  {target_name}  -->  {reference_name}  "
            f"(similarity = {similarity_score:.3f})"
        )


if __name__ == "__main__":
    main()
