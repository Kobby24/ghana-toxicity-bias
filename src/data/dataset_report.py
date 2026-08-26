from pathlib import Path
import pandas as pd


def report_jigsaw():
    path = Path("data/raw/jigsaw/train.csv")

    print("\n" + "=" * 80)
    print("JIGSAW UNINTENDED BIAS DATASET")
    print("=" * 80)

    # Only read the columns needed for the research.
    cols = [
        "id",
        "target",
        "comment_text",
        "identity_attack",
        "asian",
        "black",
        "white",
        "latino",
        "christian",
        "muslim",
        "jewish",
        "male",
        "female",
        "transgender",
    ]

    df = pd.read_csv(path, usecols=cols)

    print("Shape:", df.shape)

    print("\nTarget statistics:")
    print(df["target"].describe())

    print("\nTarget missing:", df["target"].isna().sum())

    print("\nIdentity annotation coverage:")
    identity_cols = [
        "asian",
        "black",
        "white",
        "latino",
        "christian",
        "muslim",
        "jewish",
        "male",
        "female",
        "transgender",
    ]

    for col in identity_cols:
        non_null = df[col].notna().sum()
        print(
            f"{col:15} "
            f"annotated={non_null:,} "
            f"({non_null / len(df) * 100:.2f}%)"
        )

    print("\nIdentity-positive counts:")
    for col in identity_cols:
        positive = (df[col] >= 0.5).sum()
        print(f"{col:15} {positive:,}")

    print("\nExact duplicate comments:")
    print(df["comment_text"].duplicated().sum())


def report_twi():
    path = Path("data/raw/twi_toxicity/Toxicity_dataset_M.csv")

    print("\n" + "=" * 80)
    print("TWI TOXICITY DATASET")
    print("=" * 80)

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nLabels:")
    print(df["LABEL"].value_counts())

    print("\nLabel percentages:")
    print(
        (df["LABEL"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nDuplicate comments:", df["COMMENT"].duplicated().sum())

    print("\nDuplicate comments with conflicting labels:")

    grouped = (
        df.groupby("COMMENT")["LABEL"]
        .nunique()
    )

    conflicting = grouped[grouped > 1]

    print("Count:", len(conflicting))

    if len(conflicting):
        for comment in conflicting.index:
            labels = (
                df.loc[df["COMMENT"] == comment, "LABEL"]
                .unique()
                .tolist()
            )
            print(f"  {labels}: {comment}")


def report_afrihate():
    path = Path("data/raw/afrihate/data/twi/all_data.tsv")

    print("\n" + "=" * 80)
    print("AFRIHATE TWI")
    print("=" * 80)

    df = pd.read_csv(path, sep="\t")

    print("Shape:", df.shape)

    print("\nLabels:")
    print(df["label"].value_counts())

    print("\nLabel percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nDuplicate tweets:", df["tweet"].duplicated().sum())


if __name__ == "__main__":
    report_jigsaw()
    report_twi()
    report_afrihate()
