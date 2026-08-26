import pandas as pd
from pathlib import Path
import random

OUTPUT = Path("data/interim/ghana_identity_counterfactuals_balanced.csv")

random.seed(42)

# ---------------------------------------------------------
# Identity groups
# ---------------------------------------------------------

GHANAIAN_IDENTITIES = [
    "Akan",
    "Ewe",
    "Ga",
    "Fante",
    "Asante",
    "Akuapem",
    "Akyem",
    "Kwahu",
    "Nzema",
    "Sefwi",
    "Wasa",
    "Dangme",
    "Krobo",
    "Ada",
    "Dagomba",
    "Dagbani",
    "Mamprusi",
    "Kusasi",
    "Nanumba",
    "Gonja",
    "Kasena",
    "Sisala",
]

REFERENCE_IDENTITIES = [
    "Black",
    "White",
    "Asian",
    "Latino",
    "Jewish",
]

# ---------------------------------------------------------
# Controlled templates
#
# Every identity gets EXACTLY the same templates.
# Only the identity word changes.
# ---------------------------------------------------------

TEMPLATES = {
    "positive": [
        "My neighbour is {identity} and we get along well.",
        "My friend is {identity} and they are very kind.",
        "I met someone who is {identity} and we had a pleasant conversation.",
        "The person I met is {identity} and they were very friendly.",
        "My {identity} colleague helped me yesterday.",
    ],

    "neutral": [
        "My neighbour is {identity}.",
        "My colleague is {identity}.",
        "The new student is {identity}.",
        "My friend is {identity}.",
        "The person I met is {identity}.",
    ],

    "negative_non_toxic": [
        "My {identity} neighbour was late to the meeting.",
        "I disagree with my {identity} colleague about the project.",
        "My {identity} friend forgot the appointment.",
        "The {identity} student missed the first class.",
        "My {identity} colleague made a mistake at work.",
    ],
}

# ---------------------------------------------------------
# Generate data
# ---------------------------------------------------------

rows = []

template_id = 0

for category, templates in TEMPLATES.items():

    for local_id, template in enumerate(templates, start=1):

        template_id += 1

        # Generate Ghanaian identities
        for identity in GHANAIAN_IDENTITIES:
            rows.append({
                "identity": identity,
                "identity_category": "ghanaian",
                "template_category": category,
                "template_id": local_id,
                "global_template_id": template_id,
                "template": template,
                "text": template.format(identity=identity),
                "expected_label": "non-toxic",
            })

        # Generate reference identities
        for identity in REFERENCE_IDENTITIES:
            rows.append({
                "identity": identity,
                "identity_category": "reference",
                "template_category": category,
                "template_id": local_id,
                "global_template_id": template_id,
                "template": template,
                "text": template.format(identity=identity),
                "expected_label": "non-toxic",
            })

df = pd.DataFrame(rows)

# Shuffle deterministically
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("=" * 80)
print("BALANCED COUNTERFACTUAL BENCHMARK GENERATED")
print("=" * 80)

print(f"Output: {OUTPUT}")
print(f"Rows: {len(df)}")

print("\nIdentity counts:")
print(df["identity"].value_counts().sort_index())

print("\nIdentity categories:")
print(df["identity_category"].value_counts())

print("\nTemplate categories:")
print(df["template_category"].value_counts())

print("\nExpected labels:")
print(df["expected_label"].value_counts())

print("\nTemplate balance:")
print(
    df.groupby(
        ["global_template_id", "identity_category"]
    ).size().unstack(fill_value=0)
)

print("\nFirst 20 examples:")
print(
    df[
        [
            "identity",
            "identity_category",
            "template_category",
            "template_id",
            "text",
            "expected_label",
        ]
    ].head(20).to_string(index=False)
)
