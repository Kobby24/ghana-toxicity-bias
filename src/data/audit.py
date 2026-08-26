from pathlib import Path
import pandas as pd


DATASETS = {
    "jigsaw": {
        "path": Path("data/raw/jigsaw/train.csv"),
        "separator": ",",
        "sample_rows": 1000,
    },
    "twi_toxicity": {
        "path": Path("data/raw/twi_toxicity/Toxicity_dataset_M.csv"),
        "separator": ",",
        "sample_rows": None,
    },
    "afrihate_twi": {
        "path": Path("data/raw/afrihate/data/twi/all_data.tsv"),
        "separator": "\t",
        "sample_rows": None,
    },
}


def audit_dataset(name, config):
    path = config["path"]

    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: Missing file: {path}")
        return

    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    df = pd.read_csv(
        path,
        sep=config["separator"],
        nrows=config["sample_rows"],
    )

    print(f"Rows inspected: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    for name, config in DATASETS.items():
        audit_dataset(name, config)
