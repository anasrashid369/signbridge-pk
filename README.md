# SignBridge PK

A recognizer for the static letter signs of Pakistan Sign Language (PSL), built with computer vision and machine learning. It is a learning and practice aid, not a replacement for an interpreter.

## Problem

People who sign in PSL and people who do not know it often have no shared language. Fingerspelling, signing a word letter by letter, is used for names and words without an established sign. This project recognizes individual static letter signs from a single image of one hand. Motion-based signs, words and sentences are out of scope.

## Status

Deliverable 1 (Problem Definition and Data Acquisition) is complete for the ASL baseline: download, cleaning, raw-data charts, landmark extraction and feature charts all run on it. PSL photo collection has not started yet (0 volunteers, 0 photos); it continues in Deliverable 2, where the PSL letter set is also fixed from a cited source. Models arrive in Deliverable 2 and the app in Deliverable 3.

## Repository Layout

```
signbridge-pk/
├── data_scripts/                    # All Phase 2 code
│   ├── download_data.py             # download the ASL baseline from Kaggle
│   ├── build_manifest.py            # PSL manifest with a person-wise split
│   ├── clean/                       # cleaning: quality checks, duplicates, report
│   ├── eda/                         # raw-data and feature charts
│   └── features/                    # landmark extraction, normalisation, images, labels
├── src/signbridge/                  # Shared library: config loader, logging setup
├── configs/data.yaml                # All paths, thresholds and seeds
├── tests/                           # pytest suite
├── docs/
│   ├── data_collection_protocol.md  # PSL collection protocol and consent note
│   └── report/                      # LaTeX report sources
├── reports/
│   ├── figures/                     # charts and the detection table used in the report
│   ├── cleaning_summary.csv         # image counts per class after each cleaning step
│   └── cleaning_summary_table.tex   # the same counts as a table for the report
├── data/                            # raw/ and interim/ are not in Git
│   └── processed/                   # only label_map.json (and later manifest.csv) are committed
├── models/                          # hand landmarker model and later checkpoints (not in Git)
├── app/                             # app (Deliverable 3)
├── notebooks/                       # exploration only
├── requirements.txt                 # runtime dependencies for Deliverable 1
├── requirements-dev.txt             # runtime plus pytest and ruff
├── requirements-ml.txt              # app dependencies for later deliverables
└── pyproject.toml                   # project metadata and tool settings
```

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

Download the MediaPipe hand landmarker model once (about 7.8 MB, not in Git):

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

On Windows PowerShell use `curl.exe` instead of `curl`.

Check the setup:

```bash
ruff check .
pytest
```

## Data

- **ASL baseline:** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (`grassknoted/asl-alphabet`) on Kaggle, licensed GPL 2. Its training folder has 87,000 JPEG images of 200x200 pixels in 29 folders: A-Z plus `del`, `nothing` and `space`. It is American Sign Language, used only to build and test the pipeline. Only the 26 letters are classes; `nothing` images serve as a negative test.
- **PSL photos:** to be collected by the team from consenting volunteers under `docs/data_collection_protocol.md`. None have been collected yet. They will never be committed.

Raw images, the cleaning report and the landmark cache are never committed.

## Running the Data Scripts

Run every command from the repository root. Settings come from `configs/data.yaml`, and every script accepts `--help`.

### 1. Download the ASL baseline

Needs a Kaggle account. Authenticate once with `kaggle auth login`, or set the `KAGGLE_API_TOKEN` environment variable to a token from your Kaggle account settings.

```bash
python -m data_scripts.download_data --dry-run   # show the dataset and destination, download nothing
python -m data_scripts.download_data             # download into data/raw/asl_alphabet
```

### 2. Clean the images

Flags unreadable files, missing labels, exact and near duplicates, and blurry, dark, bright or tiny images. Nothing is deleted.

```bash
python -m data_scripts.clean.clean_images --source asl --limit-per-class 100
python -m data_scripts.clean.clean_images --source asl --dir PATH/TO/SAMPLE --limit-per-class 0
python -m data_scripts.clean.clean_images --source psl
```

`--limit-per-class` cleans a seeded random sample per class (0 uses all images); `--dir` cleans another folder with one subfolder per class. Writes `data/interim/cleaning_report.csv` (not committed), `reports/cleaning_summary.csv` and `reports/cleaning_summary_table.tex`.

### 3. Draw the raw-data charts

Run after cleaning. Saves the charts to `reports/figures/`.

```bash
python -m data_scripts.eda.raw_data_eda --asl-label "ASL baseline (sample, 100 per class)"
```

`--asl-label` names the ASL data in every chart title; say there if the charts come from a sample. `--psl-sample-grid` also saves a grid of PSL photos; use it only if every volunteer shown agreed.

### 4. Create the label map

```bash
python -m data_scripts.features.labels           # writes data/processed/label_map.json (A-Z)
```

### 5. Extract hand landmarks

Needs the hand landmarker model (see Setup).

```bash
python -m data_scripts.features.extract --max-per-class 50
python -m data_scripts.features.extract --images-dir PATH/TO/SAMPLE --max-per-class 0
```

Saves `landmark_features.npz` and `landmark_detections.csv` to `data/processed/` (not committed).

### 6. Draw the feature charts

Run after extraction.

```bash
python -m data_scripts.eda.feature_eda --dataset-label "ASL baseline"
```

Saves the detection rate chart and table, the normalisation chart and the feature correlation heatmap to `reports/figures/`.

### 7. Build the PSL manifest

Needs PSL photos in `data/raw/psl`, named `{person_id}_{label}_{lighting}_{background}_{index}.jpg` as in the protocol. Splits whole volunteers into train, validation and test, and fails if a person would appear in two splits. It cannot run until PSL photos exist.

```bash
python -m data_scripts.build_manifest            # writes data/processed/manifest.csv
```

## Coming in Later Deliverables

- PSL letter set from a cited source, PSL photo collection, classifiers and MobileNetV2 training (Deliverable 2)
- App for webcam capture, top-3 predictions and landmark overlay (Deliverable 3)

## Team

- **Aliza**: Data lead (collection protocol, cleaning, raw-data charts, manifest)
- **Fatima**: Model lead (features, landmark extraction, feature charts)
- **Anas**: App and Integration lead (repository, CI, download script, report assembly, app)

## Notes

- Scope is static letter signs only.
- Deployment is not part of this project; everything runs locally.
