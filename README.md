# SignBridge PK

A recognizer for the static letter signs of Pakistan Sign Language (PSL), built with computer vision and machine learning. It is a learning and practice aid, not a replacement for an interpreter.

## Problem

People who sign in PSL and people who do not know it often have no shared language. Fingerspelling, signing a word letter by letter, is used for names and words without an established sign. This project recognizes individual static letter signs from a single image of one hand. Motion-based signs, words and sentences are out of scope.

## Status

Deliverable 1 (Problem Definition and Data Acquisition) is complete for the ASL baseline: download, cleaning, raw-data charts, landmark extraction and feature charts all run on it. PSL photo collection has not started yet (0 volunteers, 0 photos); it continues in Deliverable 2, where the PSL letter set is also fixed from a cited source. Deliverable 2 adds the MobileNetV2 CNN, the split-driven image dataset, training and tuning commands, and the Contribution Table tooling. The real CNN tuning and final run are complete on the shared split: 12 tuning trials of 3 epochs, and a final run that stopped after 9 of its 12 configured epochs with a best validation macro-F1 of 0.9686 at epoch 6 and a test macro-F1 of 0.9795, read once after the hyperparameters were frozen. These scores are optimistic because the split still contains near-duplicate images shared with the training set (see the Limitations section of the report), so they are not an estimate of accuracy on new signers. The app arrives in Deliverable 3.

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
│   ├── data/                        # split-driven dataset and image transforms
│   ├── models/                      # MobileNetV2 wrapper and landmark models
│   ├── training/                    # loops, prediction files and trial logging
│   ├── train_cnn.py                 # CNN training command
│   └── tune_cnn.py                  # CNN hyperparameter search command
├── configs/data.yaml                # All paths, thresholds and seeds
├── configs/cnn.yaml                 # CNN data, model, training and tuning settings
├── configs/modules.yaml             # module owners and path globs for the Contribution Table
├── tools/                           # subset packer, contribution table, trials to LaTeX
├── tests/                           # pytest suite
├── docs/
│   ├── data_collection_protocol.md  # PSL collection protocol and consent note
│   └── report/                      # LaTeX report sources
├── reports/
│   ├── figures/                     # charts and the detection table used in the report
│   ├── cleaning_summary.csv         # image counts per class after each cleaning step
│   └── cleaning_summary_table.tex   # the same counts as a table for the report
├── data/                            # raw/ and interim/ are not in Git
│   └── processed/                   # label_map.json, split_v1.csv and manifest.csv are committed
├── models/                          # hand landmarker model and later checkpoints (not in Git)
├── app/                             # app (Deliverable 3)
├── notebooks/                       # exploration only
├── requirements.txt                 # runtime dependencies for Deliverable 1
├── requirements-dev.txt             # runtime plus pytest and ruff
├── requirements-ml.txt              # CNN dependencies, including CPU PyTorch
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

Most tests use only the Deliverable 1 dependencies. Every test module that needs PyTorch starts with `pytest.importorskip("torch")`, so `pytest` also passes in CI, where torch is not installed. Install `requirements-ml.txt` and run `pytest` locally to include the CNN tests.

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

## Deliverable 2: CNN Training

The image classifier is MobileNetV2 from torchvision with a new dropout and linear head for the 26 ASL letters. All settings live in `configs/cnn.yaml`; paths are relative to the repository root. Training runs on CUDA when PyTorch reports a GPU, otherwise on the CPU.

Install the CNN dependencies once:

```bash
pip install -r requirements-ml.txt
```

That file includes the Deliverable 1 runtime and the pinned CPU builds of PyTorch and torchvision, so the command works on its own. Gradio is not part of this install; it returns with the Deliverable 3 app. This command is for a local CPU environment. On Colab or Kaggle, use the recipe below instead, because the `+cpu` pins would replace the preinstalled CUDA build.

The committed split contract is `data/processed/split_v1.csv`, with columns `path`, `label`, `group_id` and `split`. The `path` column is relative to the image root, for example `A/A123.jpg`; `split` is one of `train`, `val` and `test`; and class indices come from `data/processed/label_map.json`. The dataset only reads the split it is asked for.

Validate the split before training. The command checks the columns, the split values, the labels, the image paths, duplicate paths, near-duplicate groups that cross splits, and the 70/15/15 shares overall and per class:

```bash
python -m tools.check_split --split-file data/processed/split_v1.csv --image-root PATH/TO/images
```

It prints the counts and every problem, and exits non-zero when the file does not match the contract. Pass `--tolerance` and `--class-tolerance` only if the team agrees to different limits.

### Rebuild the split file

`data/processed/split_v1.csv` was built from the full ASL download (3,000 images per letter, `data/raw/asl_alphabet`) in two steps, both run from the repository root. Repeat them only to reproduce the file; the committed file is the one every model must use.

```bash
python -m data_scripts.build_split_pool
python -m signbridge.data.splits
```

The first command cleans one letter at a time: it drops unreadable images, quality failures, exact duplicates and near-duplicates, puts the remaining images into groups of look-alikes, and samples up to `images_per_class` images per letter. It writes the pool to `data/interim/split_pool.csv` (not committed) and, for every letter, how many images each step removed to `reports/split_pool_summary.csv` (committed). `--classes A B` runs only those letters and replaces only their rows. The whole run takes from a few minutes to about half an hour, depending on the computer.

The second command assigns whole groups to train, validation and test inside each letter, so no group is split across sets, and writes `data/processed/split_v1.csv`. It stops with an error if a group ends up in two splits.

The settings are in `configs/data.yaml`, section `split`, plus the top-level `seed`:

| Key | Value | Meaning |
|---|---|---|
| `images_per_class` | 400 | Images kept per letter after cleaning, drawn at random with the seed |
| `near_duplicate_distance` | 3 | Two images of a letter whose 64-bit hashes differ in at most this many bits are near-duplicates; only the first is kept. Used by the split pool only; `quality.near_duplicate_distance` is the Deliverable 1 setting |
| `group_method` | leader | `leader`: an image joins the closest earlier group leader, so groups stay small. `chain`: images linked through other images share a group, which merges a whole run of frames into one group |
| `group_distance` | 10 | Largest hash distance (in bits) at which an image joins a group |
| `train_fraction`, `val_fraction`, `test_fraction` | 0.70, 0.15, 0.15 | Share of each letter in each split |

The committed file has 10,400 images, 400 per letter, split 280 / 60 / 60 per letter (7,280 / 1,560 / 1,560 in total). It holds 2,128 groups: 1,192 in train, 459 in validation and 477 in test. Grouping by image hash reduces leakage between the splits but cannot remove it, because the ASL images have no signer labels.

### Check the GPU and estimate the run time

```bash
python -m signbridge.preflight --config configs/cnn.yaml
```

The command prints the PyTorch and torchvision versions, whether CUDA is available and the GPU name, the split counts, and the measured seconds per training batch. It then estimates one epoch, the tuning search and the final run from `configs/cnn.yaml`. It writes nothing in the repository. Add `--max-per-class N` to estimate a capped run and `--split-file` or `--image-root` when the data is elsewhere.

### Train the model

```bash
python -m signbridge.train_cnn --config configs/cnn.yaml
```

The default run writes a per-epoch history CSV with training and validation loss, accuracy, macro-F1, learning rate and seconds, keeps the checkpoint with the best validation macro-F1, and writes validation predictions. The checkpoint, history and prediction locations are set in `configs/cnn.yaml`:

- `models/cnn_best.pt` (gitignored)
- `reports/training/cnn_history.csv`
- `reports/predictions/cnn_val.npz`

Useful overrides are `--split-file`, `--image-root`, `--epochs`, `--batch-size`, `--run-name`, `--max-batches` and `--max-per-class`. A smoke test on a few batches proves that the pipeline runs end to end:

```bash
python -m signbridge.train_cnn --config configs/cnn.yaml --split-file PATH/TO/split.csv --image-root PATH/TO/images --epochs 1 --max-batches 5 --run-name smoke
```

`--max-per-class N` caps only the training split at `N` images per class. The sample is drawn per class with the config seed, and validation and test are left complete, so scores stay comparable between a capped run and a full run. Use it when CPU time is short, for example `--max-per-class 200`. A run name other than `cnn` writes `<run-name>.pt`, `<run-name>.csv` and `<run-name>_val.npz` next to the default outputs.

### Tune the hyperparameters

```bash
python -m signbridge.tune_cnn --config configs/cnn.yaml
```

The grid in `configs/cnn.yaml` has five hyperparameters, giving 32 combinations. `tune_cnn.py` shuffles that list with `random.Random(seed)` (seed 42 from the config), removes duplicate combinations, then walks the shuffled order from the start and runs at most 12 trials. A position whose parameter set and seed are already recorded in the trials CSV is skipped, so an interrupted search fills the remaining budget in the same order instead of starting a new search. Each trial runs for 3 epochs by default and appends one row to `reports/tuning/cnn_trials.csv`. Trials are ranked by validation macro-F1; ties go to the trial with fewer trainable parameters. Trial predictions are written to `reports/predictions/runs/`, which is gitignored. Use `--epochs`, `--max-trials`, `--max-batches`, `--max-per-class`, `--split-file` and `--image-root` to shorten or redirect a search. `--max-per-class` applies the same seeded training-only cap as in `train_cnn`.

After the search, turn the trials CSV into the LaTeX table used by the report:

```bash
python -m tools.trials_to_tex --csv reports/tuning/cnn_trials.csv
```

### Plot the loss and accuracy curves

Run this after the final training run, when `reports/training/cnn_history.csv` exists:

```bash
python -m tools.plot_cnn_curves
```

It writes `reports/figures/cnn_loss_curve.png` and `reports/figures/cnn_accuracy_curve.png` with the training and validation lines and the best-validation epoch marked. The figures use the Agg backend, so the command works on a headless machine. Do not commit figures made from a smoke run; commit only the figures from the final run.

### Test-set rule

The test split is never read during training, tuning or model selection. It is evaluated only when the explicit flag is passed:

```bash
python -m signbridge.train_cnn --config configs/cnn.yaml --evaluate-test
```

Run that command once, after the hyperparameters are frozen. It writes `reports/predictions/<run-name>_test.npz` and logs a warning. Every prediction file has the same four arrays: `y_true` (class indices), `y_proba` (an `n` by 26 matrix of softmax probabilities), `paths` (the split paths in order) and `class_names` (the class names in index order). The committed validation and test files are the inputs to the evaluation module.

### Package a subset for Colab or Kaggle

Only the images named in the split file are needed on the remote machine. This command writes a zip outside the repository by default path choice; the `<class>/<file>` structure is preserved:

```bash
python -m tools.pack_subset --split-file data/processed/split_v1.csv --image-root PATH/TO/images --output PATH/TO/subset.zip --splits train val test
```

`--splits` can be any non-empty subset of `train val test`. The subset is enough because the split file stores paths relative to the image root.

### Colab or Kaggle recipe

Run these commands from a fresh runtime. No notebook is needed. Colab and Kaggle already ship a CUDA build of PyTorch and torchvision, so **do not run `pip install -r requirements-ml.txt` there**: that file pins the `+cpu` wheels and would replace the preinstalled CUDA build. Install only the repository runtime and the package itself:

```bash
git clone REPOSITORY_URL signbridge-pk
cd signbridge-pk
pip install -r requirements.txt
pip install -e .
```

Dry-run notes from a Colab T4 session:

- Colab ran Python 3.13 and the pins in `requirements.txt` installed cleanly there; Kaggle accepts the same pins.
- After the two `pip install` commands, restart the runtime once (`Runtime > Restart session`), then run `%cd /content/signbridge-pk` again before the next command.
- `requirements-ml.txt` is never installed on Colab or Kaggle.

Confirm that CUDA is visible before spending time on training:

```bash
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
nvidia-smi
```

`CUDA True` and a GPU name in both commands mean the runtime is ready. If `CUDA` is `False`, stop and choose a GPU runtime; the commands still run on the CPU, but the time estimates assume a GPU.

Upload the subset zip to the runtime, or download it from the project's external storage, and unpack it. The split file and `label_map.json` come from the repository; the raw images do not. In Colab, upload `asl_subset.zip` through Drive and unzip it from the mounted folder; if `drive.mount` fails, upload the zip through the Files panel and unzip from `/content/asl_subset.zip`. Create the target folder first:

```bash
mkdir -p data/raw/asl_subset
unzip PATH/TO/subset.zip -d data/raw/asl_subset
python -m signbridge.preflight --config configs/cnn.yaml --split-file data/processed/split_v1.csv --image-root data/raw/asl_subset
```

The preflight command prints the versions, the GPU name, the split counts and the measured time per batch, and then estimates the tuning search and the final run. Continue only when it reports a GPU and the estimates fit the time left.

### Contribution table

```bash
python -m tools.contribution_table
```

This reads `configs/modules.yaml`, counts non-merge commits per author for every module with `git log --use-mailmap --no-merges` on `main`, and writes `reports/contribution_table.csv` and `reports/contribution_table.tex`. Commits that exist only on unmerged branches are not reachable from `main` and are not counted. Author aliases are grouped through `.mailmap`.

### Model weights

The trained checkpoint is not committed because it is a binary file. It is hosted externally:

**Weights:** [cnn_best.pt on Google Drive](https://drive.google.com/file/d/1bU0j8pdX5LEu9MLdJIy_xPGXsd14twWJ/view?usp=drive_link)

Download the file, place it at `models/cnn_best.pt`, and use `torch.load` with the same model definition from `signbridge.models.cnn`. Never invent or commit the link.

## Coming in Later Deliverables

- PSL photo collection and a PSL letter set confirmed from a cited source and with a PSL user
- App for webcam capture, top-3 predictions and landmark overlay (Deliverable 3)

## Team

- **Aliza**: Data lead (collection protocol, cleaning, raw-data charts, manifest)
- **Fatima**: Model lead (features, landmark extraction, feature charts)
- **Anas**: App and Integration lead (repository, CI, download script, report assembly, app)

## Notes

- Scope is static letter signs only.
- Deployment is not part of this project; everything runs locally.
