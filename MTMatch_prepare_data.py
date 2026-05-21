import glob
import os
import random

import pandas as pd


def prepare_mtmatch_pretrain_data(csv_dir, output_dir, max_cells=128):
    """Serialize all CSV columns in a directory into MTMatch pretraining text."""
    os.makedirs(output_dir, exist_ok=True)

    all_serialized_columns = []
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files under {csv_dir}")

    for file_path in csv_files:
        try:
            dataframe = pd.read_csv(file_path)
            for column_name in dataframe.columns:
                cells = dataframe[column_name].dropna().astype(str).tolist()
                if len(cells) > max_cells:
                    cells = random.sample(cells, max_cells)
                serialized_text = f"[COL] {column_name} " + " ".join([f"[VAL] {cell}" for cell in cells])
                all_serialized_columns.append(serialized_text)
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")

    if not all_serialized_columns:
        raise RuntimeError(f"No serialized columns were produced from {csv_dir}")

    train_no_label_path = os.path.join(output_dir, "train_no_label.txt")
    with open(train_no_label_path, "w", encoding="utf-8") as fout:
        for line in all_serialized_columns:
            fout.write(line.replace("\n", " ").replace("\r", " ") + "\n")

    print(f"Wrote {len(all_serialized_columns)} serialized columns to {train_no_label_path}")

    dummy_line = f"{all_serialized_columns[0]}\t{all_serialized_columns[0]}\t1\n"
    for split in ["train.txt", "valid.txt", "test.txt", "tableA.txt", "tableB.txt"]:
        with open(os.path.join(output_dir, split), "w", encoding="utf-8") as fout:
            fout.write(dummy_line)


if __name__ == "__main__":
    prepare_mtmatch_pretrain_data(
        csv_dir="data/Tablematch/",
        output_dir="data/Tablematch/mtmatch_pretrain_data/",
    )
