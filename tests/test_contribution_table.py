"""Tests for tools/contribution_table.py on a temporary git repository."""

import csv
import os
import subprocess
from pathlib import Path

import yaml

from tools.contribution_table import main, module_for_path


def git(
    repo: Path, *args: str, author: str = "Test Author", email: str = "test@example.com"
) -> None:
    """Run git in a temporary repository with a fixed identity."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": author,
        "GIT_COMMITTER_EMAIL": email,
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def test_counts_commits_per_module(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    source = repo / "src"
    source.mkdir()
    (source / "cnn.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "src/cnn.py")
    git(repo, "commit", "-m", "add cnn")
    config = tmp_path / "modules.yaml"
    config.write_text(
        yaml.safe_dump(
            {"modules": {"cnn": {"owner": "Anas", "paths": ["src/**"]}}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "table.csv"
    tex_path = tmp_path / "table.tex"
    code = main(
        [
            "--config",
            str(config),
            "--repo",
            str(repo),
            "--csv",
            str(csv_path),
            "--tex",
            str(tex_path),
        ]
    )
    assert code == 0
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"module": "cnn", "owner": "Anas", "author": "Test Author", "commits": "1"}]
    assert "Test Author (1)" in tex_path.read_text(encoding="utf-8")


def test_dot_directory_glob_keeps_its_dot() -> None:
    modules = {"project": {"owner": "Anas", "paths": [".github/**"]}}
    assert module_for_path(".github/workflows/ci.yml", modules) == "project"


def test_most_specific_module_wins(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    training = repo / "src" / "signbridge" / "training"
    training.mkdir(parents=True)
    (training / "loop.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "src/signbridge/training/loop.py")
    git(repo, "commit", "-m", "add loop", author="Anas Rashid")
    (training / "landmark_trials.py").write_text("y = 2\n", encoding="utf-8")
    git(repo, "add", "src/signbridge/training/landmark_trials.py")
    git(repo, "commit", "-m", "add landmark trials", author="Fatima Asif")

    config = tmp_path / "modules.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "modules": {
                    "cnn_training": {
                        "owner": "Anas",
                        "paths": ["src/signbridge/training/**"],
                    },
                    "landmark_models": {
                        "owner": "Fatima",
                        "paths": ["src/signbridge/training/landmark_trials.py"],
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "table.csv"
    code = main(
        [
            "--config",
            str(config),
            "--repo",
            str(repo),
            "--csv",
            str(csv_path),
            "--tex",
            str(tmp_path / "table.tex"),
        ]
    )
    assert code == 0
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"module": "cnn_training", "owner": "Anas", "author": "Anas Rashid", "commits": "1"},
        {"module": "landmark_models", "owner": "Fatima", "author": "Fatima Asif", "commits": "1"},
    ]
