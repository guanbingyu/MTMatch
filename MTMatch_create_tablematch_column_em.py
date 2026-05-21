import argparse
import csv
import os
import random
import shutil
from typing import Dict, List, Tuple


def read_table_columns(
    csv_path: str,
    max_rows: int,
) -> Dict[str, List[str]]:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV file: {csv_path}")

        columns: Dict[str, List[str]] = {
            column_name: [] for column_name in reader.fieldnames
        }
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break
            for column_name, value in row.items():
                if value is None:
                    continue
                columns[column_name].append(str(value))
    return columns


def build_column_texts(
    csv_path: str,
    max_rows: int,
    max_cells: int,
) -> Tuple[List[str], List[str]]:
    columns = read_table_columns(csv_path, max_rows=max_rows)
    names: List[str] = list(columns.keys())
    texts: List[str] = []
    for name in names:
        values = columns[name]
        tokens: List[str] = []
        for value_text in values:
            stripped = value_text.strip()
            if not stripped:
                continue
            tokens.append(stripped)
            if len(tokens) >= max_cells:
                break
        texts.append(" ".join(tokens))
    return names, texts


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line)
            if not line.endswith("\n"):
                fout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create MTMatch EM-style datasets for Tablematch column matching. "
            "Each column in the reference/target CSV becomes one entity in tableA.txt/tableB.txt, "
            "and pseudo labels are generated from header name equality."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference CSV file (mapped to tableA.*).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target CSV file (mapped to tableB.*).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Tablematch_columns",
        help="EM task name under data/em/ to store generated files.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=50000,
        help="Maximum number of rows to read from each CSV.",
    )
    parser.add_argument(
        "--max_cells_per_column",
        type=int,
        default=128,
        help="Maximum number of cells used to serialize each column.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling train/valid/test splits.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = os.path.join("data", "em", args.task)
    os.makedirs(output_dir, exist_ok=True)

    ref_names, ref_texts = build_column_texts(
        args.reference, max_rows=args.max_rows, max_cells=args.max_cells_per_column
    )
    tgt_names, tgt_texts = build_column_texts(
        args.target, max_rows=args.max_rows, max_cells=args.max_cells_per_column
    )

    write_lines(
        os.path.join(output_dir, "tableA.txt"),
        [text for text in ref_texts],
    )
    write_lines(
        os.path.join(output_dir, "tableB.txt"),
        [text for text in tgt_texts],
    )

    write_lines(
        os.path.join(output_dir, "tableA_columns.txt"),
        [name for name in ref_names],
    )
    write_lines(
        os.path.join(output_dir, "tableB_columns.txt"),
        [name for name in tgt_names],
    )

    unique_entities = []
    seen = set()
    for text in ref_texts + tgt_texts:
        if text not in seen:
            seen.add(text)
            unique_entities.append(text)
    write_lines(
        os.path.join(output_dir, "train_no_label.txt"),
        unique_entities,
    )

    pairs: List[Tuple[str, str, int]] = []
    for left_name, left_text in zip(ref_names, ref_texts):
        left_key = left_name.strip().lower()
        for right_name, right_text in zip(tgt_names, tgt_texts):
            right_key = right_name.strip().lower()
            label = int(left_key == right_key)
            pairs.append((left_text, right_text, label))

    random.shuffle(pairs)
    num_pairs = len(pairs)
    num_train = int(num_pairs * 0.8)
    num_valid = int(num_pairs * 0.1)
    num_test = num_pairs - num_train - num_valid

    train_pairs = pairs[:num_train]
    valid_pairs = pairs[num_train : num_train + num_valid]
    test_pairs = pairs[num_train + num_valid :]

    def format_pairs(subset: List[Tuple[str, str, int]]) -> List[str]:
        return [f"{left}\t{right}\t{label}" for left, right, label in subset]

    write_lines(os.path.join(output_dir, "train.txt"), format_pairs(train_pairs))
    write_lines(os.path.join(output_dir, "valid.txt"), format_pairs(valid_pairs))
    write_lines(os.path.join(output_dir, "test.txt"), format_pairs(test_pairs))

    all_pairs_lines = format_pairs(pairs)
    write_lines(os.path.join(output_dir, "all_pairs.txt"), all_pairs_lines)

    shutil.copyfile(
        args.reference, os.path.join(output_dir, "tableA.csv")
    )
    shutil.copyfile(
        args.target, os.path.join(output_dir, "tableB.csv")
    )

    print(f"Wrote EM-style data for task '{args.task}' to {output_dir}")
    print(f"  #columns tableA = {len(ref_names)}, tableB = {len(tgt_names)}")
    print(f"  #pairs = {num_pairs} (train={num_train}, valid={num_valid}, test={num_test})")


if __name__ == "__main__":
    main()
