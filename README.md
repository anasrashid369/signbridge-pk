# SignBridge PK

A real-time recognizer for the static letters of the Pakistan Sign Language (PSL) alphabet, built with computer vision and machine learning.

## Problem

This project recognizes individual letter signs of the Pakistan Sign Language alphabet from webcam images. It scopes the problem to static letter signs and builds a dataset, features, and models for that task. Motion-based signs and full sentences are out of scope.

## Status

Deliverable 1 (Problem Definition and Data Acquisition) is in progress. The web app and models arrive in later deliverables.

## Repository Layout

```
signbridge-pk/
├── data_scripts/          # All Phase 2 code: collect/, clean/, features/, eda/, download_data.py
├── src/signbridge/        # Reusable library code (config loader, features, models, inference)
├── configs/               # YAML settings: paths, thresholds, seeds
├── tests/                 # pytest suite
├── docs/
│   ├── proposal/          # LaTeX proposal
│   └── report/            # LaTeX final report
├── reports/               # Metrics, figures, bias analysis outputs
├── data/
│   ├── raw/               # Downloaded or collected images (not in Git)
│   ├── interim/           # Intermediate files (not in Git)
│   └── processed/         # manifest.csv and label_map.json are tracked; the rest is not
├── models/                # Model checkpoints (weights not in Git)
├── notebooks/             # Exploration only; imports from src/
├── requirements.txt       # Runtime dependencies for Deliverable 1
├── requirements-dev.txt   # Test and lint tools
├── requirements-ml.txt    # Model training and app dependencies (later deliverables)
└── pyproject.toml         # Project metadata and tool configuration
```

Phase 2 entry-point scripts live in `data_scripts/`. The shared loader in `src/signbridge/config.py` is a library module, so it stays under `src/`.

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

`requirements-dev.txt` includes the runtime dependencies plus pytest and ruff. `pip install -e .` makes the `signbridge` package importable from the scripts.

Check the setup:

```bash
pytest
ruff check .
```

## Data

- **Base dataset (ASL, baseline only):** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (`grassknoted/asl-alphabet`) on Kaggle, licensed GPL 2. It has 87,000 training images of 200x200 pixels in 29 classes (A-Z, SPACE, DELETE, NOTHING). It is used only as a baseline; it is American Sign Language, not PSL. Downloaded by `data_scripts/download_data.py`.
- **PSL dataset:** collected by the team from consenting volunteers. It is not in Git. Its hosting link will be added here when published.

Raw images are never committed. Only `data/processed/manifest.csv` and `data/processed/label_map.json` are tracked.

## Running the Data Scripts

Run every script from the repository root. Settings (paths, dataset slug, thresholds, seed) come from `configs/data.yaml`, and every script accepts `--help`.

### Download the ASL baseline dataset

Needs a Kaggle account. Authenticate once with `kaggle auth login`, or set the `KAGGLE_API_TOKEN` environment variable to a token from your Kaggle account settings.

```bash
python data_scripts/download_data.py --dry-run   # show the dataset and destination, download nothing
python data_scripts/download_data.py             # download into data/raw/asl_alphabet
```

Options: `--dataset owner/name` overrides the Kaggle slug, `--dest PATH` overrides the download folder, and `--config PATH` uses a different config file. After the download, the script logs where to place the team's PSL images (`data/raw/psl`).

### Build the PSL manifest

Needs the team's PSL photos in `data/raw/psl`, named `{person_id}_{label}_{lighting}_{background}_{index}.jpg` (see `docs/data_collection_protocol.md`). Splits whole volunteers into train, validation and test, and fails if a person would appear in two splits.

```bash
python data_scripts/build_manifest.py            # writes data/processed/manifest.csv
```

### Clean the images

Flags unreadable files, missing labels, exact and near duplicates, and blurry, dark, bright or tiny images. Nothing is deleted. Run it once per dataset; `--limit-per-class 500` cleans a random sample per class for a quick run.

```bash
python -m data_scripts.clean.clean_images --source asl --limit-per-class 500
python -m data_scripts.clean.clean_images --source psl
```

Writes `data/interim/cleaning_report.csv` (one row per image, not committed), `reports/cleaning_summary.csv` (counts per class after each step) and `reports/cleaning_summary_table.tex` (table for the report).

### Draw the raw-data charts

Run after cleaning. Saves the figures for the report to `reports/figures/`. Add `--psl-sample-grid` only if every volunteer shown agreed to it.

```bash
python -m data_scripts.eda.raw_data_eda
python -m data_scripts.eda.raw_data_eda --asl-label "ASL baseline (sample, 100 per class)"   # if you cleaned a sample
```

`--asl-label` sets how the ASL data is named in every figure title. Use it whenever the ASL charts come from a sample and not the full download.

## Coming in Later Deliverables

- Feature extraction, classifiers and MobileNetV2 training (Deliverable 2)
- Gradio web app for webcam capture, top-3 predictions and landmark overlay (Deliverable 3)

## Team

- **Aliza**: Data lead (collection, cleaning, bias analysis)
- **Fatima**: Model lead (features, training, evaluation)
- **Anas**: App and Integration lead (repository, CI, download script, app, report assembly)

## Notes

- Scope is static letter signs only.
- Deployment is excluded; the app runs locally.
   ### Extract hand landmarks

   Download the MediaPipe hand landmarker model once (it is not in Git):

```
   curl.exe -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

   Scripts inside subfolders are run as modules from the repository root:

```
   python -m data_scripts.features.extract --help
```