from pathlib import Path
import pandas as pd


PATH = Path(
    "data/interim/ghana_identity_counterfactuals.csv"
)


def main():

    df = pd.read_csv(PATH)

    print("=" * 80)
    print("BENCHMARK VALIDATION")
    print("=" * 80)

    errors = []

    # ---------------------------------------------------------
    # 1. Missing values
    # ---------------------------------------------------------

    required = [
        "identity",
        "identity_category",
        "template_category",
        "template_id",
        "template",
        "text",
        "expected_label",
    ]

    for column in required:

        count = df[column].isna().sum()

        if count:
            errors.append(
                f"{column}: {count} missing values"
            )

    # ---------------------------------------------------------
    # 2. Duplicate sentences
    # ---------------------------------------------------------

    duplicates = df["text"].duplicated().sum()

    print(
        f"\nDuplicate sentences: {duplicates}"
    )

    if duplicates:
        errors.append(
            f"{duplicates} duplicate sentences"
        )

    # ---------------------------------------------------------
    # 3. Incorrect expected labels
    # ---------------------------------------------------------

    unexpected_labels = (
        df.loc[
            df["expected_label"] != "non-toxic"
        ]
    )

    print(
        "Unexpected labels:",
        len(unexpected_labels)
    )

    # ---------------------------------------------------------
    # 4. Known article problems
    # ---------------------------------------------------------

    bad_patterns = [
        "a Akan",
        "a Ewe",
        "a Asante",
        "a Akuapem",
        "a Akyem",
        "a Nzema",
        "a Sefwi",
        "a Ewe",
    ]

    print("\nArticle/grammar checks:")

    for pattern in bad_patterns:

        matches = df[
            df["text"].str.contains(
                pattern,
                case=False,
                regex=False,
                na=False,
            )
        ]

        if len(matches):

            print(
                f"  FOUND '{pattern}': "
                f"{len(matches)}"
            )

            errors.append(
                f"Bad article pattern: {pattern}"
            )

    # ---------------------------------------------------------
    # 5. Identity/template coverage
    # ---------------------------------------------------------

    print("\nIdentity coverage:")

    coverage = (
        df.groupby(
            [
                "identity_category",
                "identity",
            ]
        )
        .size()
    )

    print(coverage.to_string())

    # ---------------------------------------------------------
    # 6. Template coverage
    # ---------------------------------------------------------

    print("\nTemplate coverage:")

    template_counts = (
        df.groupby(
            [
                "identity",
                "template_category",
            ]
        )
        .size()
    )

    print(
        template_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 80)

    if errors:

        print("VALIDATION FAILED")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    else:

        print("VALIDATION PASSED")
        print(
            "No structural or known grammar problems detected."
        )


if __name__ == "__main__":
    main()
