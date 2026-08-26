from pathlib import Path
import yaml
import pandas as pd


CONFIG_PATH = Path("configs/benchmark.yaml")
OUTPUT_PATH = Path(
    "data/interim/ghana_identity_counterfactuals.csv"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_benchmark(config):

    ghanaian = (
        config["identity_groups"]
        ["ghanaian_ethnicity"]
        ["identities"]
    )

    reference = (
        config["identity_groups"]
        ["reference"]
        ["identities"]
    )

    rows = []

    for identity_category, identities in [
        ("ghanaian", ghanaian),
        ("reference", reference),
    ]:

        for identity in identities:

            for category, templates in config["templates"].items():

                for template_id, template in enumerate(
                    templates,
                    start=1
                ):

                    text = template.format(
                        identity=identity
                    )

                    rows.append(
                        {
                            "identity": identity,
                            "identity_category": identity_category,
                            "template_category": category,
                            "template_id": template_id,
                            "template": template,
                            "text": text,
                            "expected_label": "non-toxic",
                        }
                    )

    return pd.DataFrame(rows)


def main():

    config = load_config()

    df = generate_benchmark(config)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 80)
    print("BENCHMARK GENERATED")
    print("=" * 80)

    print("Output:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("\nIdentity counts:")
    print(
        df["identity"]
        .value_counts()
        .to_string()
    )

    print("\nIdentity categories:")
    print(
        df["identity_category"]
        .value_counts()
        .to_string()
    )

    print("\nTemplate categories:")
    print(
        df["template_category"]
        .value_counts()
        .to_string()
    )

    print("\nExpected labels:")
    print(
        df["expected_label"]
        .value_counts()
        .to_string()
    )

    print("\nExamples:")
    print(
        df.sample(
            min(20, len(df)),
            random_state=42
        )[[
            "identity",
            "identity_category",
            "template_category",
            "template_id",
            "text",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
