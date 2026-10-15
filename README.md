# ghana-toxicity-bias

Does an off-the-shelf toxicity classifier score **benign** sentences more
harshly when they mention a Ghanaian ethnic group than when they mention a
reference identity?

The project builds a counterfactual benchmark: identical, non-toxic sentence
templates in which only the identity word changes. Every sentence is expected
to be non-toxic, so any score difference between identity groups is
attributable to the identity term. The model evaluated is
[`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert).

> **Status: work in progress.** Read [Known issues](#known-issues-read-before-citing-any-number)
> before using any numbers from `data/results/`.

## Benchmark design

| Item | Value |
|------|-------|
| Ghanaian identities (22) | Akan, Ewe, Ga, Fante, Asante, Akuapem, Akyem, Kwahu, Nzema, Sefwi, Wasa, Dangme, Krobo, Ada, Dagomba, Dagbani, Mamprusi, Kusasi, Nanumba, Gonja, Kasena, Sisala |
| Reference identities (5) | Black, White, Asian, Latino, Jewish |
| Templates | 15 = 5 `positive` + 5 `neutral` + 5 `negative_non_toxic` |
| Rows | 27 identities x 15 templates = **405** (330 Ghanaian, 75 reference) |
| Expected label | `non-toxic` for every row |
| Seed | 42 (shuffle in `generate_balanced.py`) |

Raw material, if you want to extend it: `configs/benchmark.yaml` (the 0.2 spec)
and `configs/identities.yaml` (candidate identities, not yet linguistically
validated).

## Repository layout

```
configs/
  benchmark.yaml            benchmark spec (identities, templates, planned tests)
  identities.yaml           candidate identity list
src/
  benchmark/
    generate.py             benchmark from configs/benchmark.yaml (original version)
    generate_balanced.py    balanced benchmark used by the notebook (templates hard-coded)
    validate.py             structural/grammar checks (runs on generate.py output)
  data/
    audit.py                schema/missing/duplicate audit of the raw datasets
    dataset_report.py       label and identity-coverage report of the raw datasets
notebooks/
  03_model_evaluation.ipynb scoring with toxic-bert + all statistics and figures
data/
  raw/                      NOT in git (.gitignore); you download this (see below)
  interim/                  NOT in git; created by the generators
  results/                  CSV tables and figures saved by the notebook
literature/
  literature_matrix.csv     reading list
```

## Requirements

- Python 3.12 (3.12.1 was used)
- About 2 GB free disk for PyTorch plus the model weights (~420 MB download)
- Internet access to `huggingface.co` the first time the model loads
- A GPU is **not** required: 405 short sentences score in seconds on CPU
- A Kaggle account, only for the optional raw-data steps

## Setup

```bash
git clone https://github.com/Kobby24/ghana-toxicity-bias.git
cd ghana-toxicity-bias

python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below are run **from the repository root**, because the scripts
use relative paths (`configs/...`, `data/...`).

## Reproduction

### Step 1: generate the benchmark (no downloads needed)

```bash
python src/benchmark/generate_balanced.py
```

Writes `data/interim/ghana_identity_counterfactuals_balanced.csv` (405 rows,
8 columns). This is the file the notebook scores. It is deterministic: running
it again gives the same row order.

Optional, the config-driven variant and its checks:

```bash
python src/benchmark/generate.py     # data/interim/ghana_identity_counterfactuals.csv
python src/benchmark/validate.py     # expect: VALIDATION PASSED
```

`validate.py` only validates the output of `generate.py`, not the balanced
file. The two generators use slightly different templates (see
[Known issues](#known-issues-read-before-citing-any-number)).

### Step 2: score the benchmark and run the statistics

```bash
jupyter lab notebooks/03_model_evaluation.ipynb
```

Launch it from the repo root or from `notebooks/`; cell 0 handles both. The
first run downloads `unitary/toxic-bert` from the Hugging Face Hub.

**Do not use "Run All".** The notebook was developed interactively and its
cells depend on execution order (see Known issues). To reproduce the
results that are consistent with the model's real output, run these cells in
order (0-based indexes, counting every cell):

| Order | Cells | What they do |
|-------|-------|--------------|
| 1 | 0-3 | Load the benchmark and check its counterfactual structure |
| 2 | 7 | Load `unitary/toxic-bert` through the `pipeline` API |
| 3 | 9 | Score all 405 sentences (`predictions`) |
| 4 | 52, 53, 54, 55 | Build `results` from `predictions`; identity ranking; template categories; Ghanaian vs reference |
| 5 | 56 | Save `identity_ranking.csv`, `template_category_results.csv`, `ghanaian_vs_reference.csv`, `model_predictions.csv` |
| 6 | 58 | Welch t-test, Mann-Whitney U, Cohen's d |
| 7 | 59-63, 69, 70 | Paired per-template analysis: paired t-test, Wilcoxon, Cohen's dz, bootstrap CI (seed 42) |
| 8 | 71-75 | Plots |

Step 2 needs only `data/interim/ghana_identity_counterfactuals_balanced.csv`
from Step 1. No Kaggle data is used by the scoring cells.

### Expected output (checks against the committed results)

If your run matches, `ghanaian_vs_reference.csv` should show:

| identity_category | n | mean toxicity | median | max |
|---|---|---|---|---|
| ghanaian | 330 | 0.00097 | 0.00076 | 0.0123 |
| reference | 75 | 0.0268 | 0.0026 | 0.582 |

and the paired analysis (15 templates) should report a mean reference-minus-
Ghanaian gap of 0.0258, bootstrap 95% CI [0.0115, 0.0444], paired t-test
p = 0.0116, Wilcoxon p = 6.1e-05, Cohen's dz = 0.75, and reference > Ghanaian
in 15/15 templates. The highest-scoring identity is `Black` (mean 0.117). Small
last-digit differences across `torch` and `transformers` versions are normal.

I could not run the model-dependent steps myself (no Hugging Face access in my
environment). The numbers above are taken from the committed notebook output
and CSVs, and Step 1 and `validate.py` were run and verified.

### Step 3 (optional): raw-dataset audit

`src/data/audit.py` and `src/data/dataset_report.py` expect three datasets
under `data/raw/`, which is git-ignored:

| Dataset | Expected path |
|---------|---------------|
| Jigsaw Unintended Bias in Toxicity Classification | `data/raw/jigsaw/train.csv` |
| Twi toxicity dataset | `data/raw/twi_toxicity/Toxicity_dataset_M.csv` |
| AfriHate (Twi) | `data/raw/afrihate/data/twi/all_data.tsv` |

The repo does not record where the Twi toxicity dataset came from, so fill in
its source here: `<TODO: source URL>`. Jigsaw is available on Kaggle
(`jigsaw-unintended-bias-in-toxicity-classification`; accept the competition
rules first). AfriHate is published by its authors on GitHub/Hugging Face.

Kaggle credentials go **outside** the repo:

```bash
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -p data/raw/jigsaw
unzip data/raw/jigsaw/*.zip -d data/raw/jigsaw
```

Then:

```bash
python src/data/audit.py
python src/data/dataset_report.py
```

## Known issues (read before citing any number)

These come from reading the code and the saved notebook output.

1. **Two incompatible score definitions are mixed in the notebook.**
   `toxic-bert` is a *multi-label* model. The `pipeline` call (cell 9)
   scores each label independently, giving toxic scores around 0.001 for
   these benign sentences. Cells 14-17, 25 and 40 instead apply
   `softmax` across all six labels, which forces the scores to sum to 1 and
   produces values around 0.5 for every sentence. Those softmax scores are
   not valid toxicity probabilities, and they are why cell 96 reports "405/405
   predicted toxic, 0% accuracy". Sigmoid (what `pipeline` does) is the
   correct choice for this model.

2. **Committed result files come from both versions.** In `data/results/`,
   `identity_ranking.csv`, `template_category_results.csv` and
   `ghanaian_vs_reference.csv` come from the pipeline scores (Ghanaian mean
   0.0010 vs reference 0.0268). `identity_toxicity_summary.csv`,
   `group_toxicity_summary.csv`, `template_toxicity_summary.csv`,
   `identity_bias_metrics.csv`, `template_bias_metrics.csv`,
   `template_bias_gaps.csv`, and the four PNGs in `figures/` come from the
   softmax scores (Ghanaian mean 0.527 vs reference 0.629). The
   `notebooks/data/results/` folder holds older softmax outputs as well. Treat
   the softmax-based files as stale until they are regenerated.

3. **Twi evaluation is wrong (cell 78).** It reads `probabilities[:, 1]`
   as "toxic", but index 1 is `severe_toxic` in this model (index 0 is
   `toxic`). No Twi results are saved, but anything derived from that cell
   should be discarded. `toxic-bert` is also an English-only model.

4. **Run All fails.** Cells 35 and 37 reference columns/variables
   (`toxicity_score` before it exists, `TOXICITY_COL`) that are never defined
   in a clean run, and later cells overwrite `results` with the softmax version.
   Results also depend on execution order.

5. **Two generators, two template sets.** `generate.py` (via
   `benchmark.yaml`) uses templates like "My colleague is {identity} and we
   work well together." `generate_balanced.py` hard-codes different ones
   (e.g. "My {identity} colleague helped me yesterday.") and ignores the
   YAML. The notebook uses the balanced file; `validate.py` checks the other.

6. **Hard-coded paths.** Cells 35 and 52+ use an absolute Codespaces path
   (`/workspaces/ghana-toxicity-bias/...`) or `../data/...`, so they only
   work when the notebook is launched from `notebooks/` or in that Codespace.

7. **Planned but not implemented.** `benchmark.yaml` lists a false-positive
   rate, an `identity_effect` metric and per-sentence duplicate prevention;
   the code does not compute these.

8. **Credential leak.** `kaggle.json` is committed with a live API token.
   Expire it at kaggle.com -> Settings -> API, create a new one, and run
   `git rm --cached kaggle.json && echo kaggle.json >> .gitignore`. The old
   token remains in git history until it is expired.

## Suggested next steps

- Move the scoring logic out of the notebook into `src/evaluation/` as one
  script (pipeline scores, `top_k=None` to also keep all six labels) that
  writes every table and figure, then delete the softmax cells.
- Regenerate all of `data/results/` from that single script.
- Add a `LICENSE` and a description to the GitHub repo.

