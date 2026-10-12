# SignBridge PK

Real-time Pakistan Sign Language (PSL) alphabet recognizer using computer vision and deep learning.

## Overview

SignBridge PK is a web-based application that recognizes static hand gestures representing letters of the Pakistan Sign Language alphabet. The system uses MediaPipe for hand landmark detection and combines classical ML and deep learning models for classification.

**Status:** Development Phase 1 (Problem Definition & Data Acquisition)

## Tech Stack

- **Python 3.11**
- **PyTorch** — neural network training and inference
- **MediaPipe** — hand landmark detection
- **OpenCV** — image processing
- **Gradio** — web interface
- **scikit-learn** — classical ML baselines (Random Forest, SVM)

## Project Structure

```
signbridge-pk/
├── app/                          # Gradio web interface
├── configs/                       # YAML configs for training, paths
├── data/
│   └── processed/               # Processed dataset (not in Git)
├── models/                        # Trained model checkpoints
├── reports/                       # Metrics, figures, bias analysis
├── scripts/                       # Data collection, download, training scripts
├── src/signbridge/
│   ├── data/                     # Dataset loading, augmentation
│   ├── features/                 # Landmark extraction, normalization
│   ├── inference/                # Prediction wrapper
│   ├── models/                   # Model architectures
│   └── evaluation/               # Metrics, bias analysis
├── tests/                         # pytest test suite
├── docs/
│   ├── proposal/                 # LaTeX proposal
│   └── report/                   # LaTeX final report
├── README.md
├── requirements.txt
├── pyproject.toml
└── .github/workflows/ci.yml      # GitHub Actions CI/CD
```

## Setup

### Prerequisites
- Python 3.11 or later
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anasrashid369/signbridge-pk.git
cd signbridge-pk
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
python app/app.py
```

The app will launch on `http://localhost:7860` by default.

## Project Phases

- **Phase 1:** Problem Definition
- **Phase 2:** Data Acquisition & Setup
- **Phase 3:** Model Implementation & Training
- **Phase 4:** Evaluation & Comparative Analysis
- **Phase 5:** Output (web app)
- **Phase 6:** Documentation & Final Report

## Team

- **Aliza** — Data Lead
- **Fatima** — Model Lead
- **Anas** — App & DevOps Lead

## License

TBD

## Notes

- This project focuses on static letter recognition only.
- Deployment is excluded; the app runs locally.
- Raw dataset images and large model files are not stored in Git.
