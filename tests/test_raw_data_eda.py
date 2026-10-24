"""Tests for data_scripts/eda/raw_data_eda.py."""

from pathlib import Path

import pandas as pd
import pytest

from data_scripts.clean import clean_images
from data_scripts.eda import raw_data_eda
from tests.test_clean_images import make_asl_folder, make_psl_folder, write_config

FIGURES = [
    "class_distribution.png",
    "image_size_distribution.png",
    "images_per_person.png",
    "blur_scores.png",
    "brightness_distribution.png",
    "asl_sample_grid.png",
]


def clean_both(tmp_path: Path) -> Path:
    """Run the cleaning script on small ASL and PSL folders and return the config path."""
    make_asl_folder(tmp_path)
    make_psl_folder(tmp_path)
    config = write_config(tmp_path)
    for source in ("asl", "psl"):
        assert clean_images.main(["--source", source, "--config", str(config)]) == 0
    return config


def add_figures_dir(config: Path, tmp_path: Path) -> None:
    """Add the figures folder, which the cleaning tests' config does not need."""
    text = config.read_text(encoding="utf-8")
    config.write_text(
        text.replace("paths:\n", f"paths:\n  figures_dir: {tmp_path / 'figures'}\n"),
        encoding="utf-8",
    )


def test_run_saves_every_figure(tmp_path: Path) -> None:
    config = clean_both(tmp_path)
    add_figures_dir(config, tmp_path)
    assert raw_data_eda.main(["--config", str(config)]) == 0
    for name in FIGURES:
        assert (tmp_path / "figures" / name).stat().st_size > 0
    assert not (tmp_path / "figures" / "psl_sample_grid.png").exists()


def test_psl_sample_grid_is_only_saved_when_asked(tmp_path: Path) -> None:
    config = clean_both(tmp_path)
    add_figures_dir(config, tmp_path)
    assert raw_data_eda.main(["--config", str(config), "--psl-sample-grid"]) == 0
    assert (tmp_path / "figures" / "psl_sample_grid.png").stat().st_size > 0


def test_run_with_only_asl_skips_the_person_chart(tmp_path: Path) -> None:
    make_asl_folder(tmp_path)
    config = write_config(tmp_path)
    assert clean_images.main(["--source", "asl", "--config", str(config)]) == 0
    add_figures_dir(config, tmp_path)
    assert raw_data_eda.main(["--config", str(config)]) == 0
    assert not (tmp_path / "figures" / "images_per_person.png").exists()
    assert (tmp_path / "figures" / "class_distribution.png").exists()


def test_main_without_a_report_returns_error(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    add_figures_dir(config, tmp_path)
    assert raw_data_eda.main(["--config", str(config)]) == 1


def test_load_report_rejects_an_empty_report(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    path.write_text(",".join(clean_images.REPORT_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        raw_data_eda.load_report(path)


def test_first_images_per_class_picks_one_readable_image_per_label(tmp_path: Path) -> None:
    clean_both(tmp_path)
    report = raw_data_eda.load_report(tmp_path / "interim/cleaning_report.csv")
    examples = raw_data_eda.first_images_per_class(report, "asl")
    assert [label for label, _ in examples] == ["A", "B"]
    assert all(path.endswith(".png") for _, path in examples)


def test_plot_blur_scores_needs_positive_scores() -> None:
    report = pd.DataFrame({"source": ["asl"], "blur_score": [0.0]})
    with pytest.raises(ValueError):
        raw_data_eda.plot_blur_scores(report, 100.0, Path("unused.png"))


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as info:
        raw_data_eda.parse_args(["--help"])
    assert info.value.code == 0


def capture_titles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace save_figure so the test can read the titles instead of saving a PNG."""
    titles: list[str] = []

    def fake_save(figure, path) -> None:  # noqa: ANN001
        titles.extend(ax.get_title(loc="left") for ax in figure.axes if ax.get_title(loc="left"))
        if figure._suptitle is not None:
            titles.append(figure._suptitle.get_text())

    monkeypatch.setattr(raw_data_eda, "save_figure", fake_save)
    return titles


def test_asl_label_appears_in_every_asl_figure_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_both(tmp_path)
    report = raw_data_eda.load_report(tmp_path / "interim/cleaning_report.csv")
    label = "ASL baseline (sample, 100 per class)"
    names = {"asl": label, "psl": "PSL"}
    titles = capture_titles(monkeypatch)
    raw_data_eda.plot_class_distribution(report, tmp_path / "a.png", names)
    raw_data_eda.plot_image_sizes(report, tmp_path / "b.png", names)
    raw_data_eda.plot_blur_scores(report, 100.0, tmp_path / "c.png", names)
    raw_data_eda.plot_brightness(report, 40, 220, tmp_path / "d.png", names)
    raw_data_eda.plot_sample_grid(report, "asl", tmp_path / "e.png", names=names)
    asl_titles = [title for title in titles if "PSL" not in title]
    assert len(asl_titles) >= 5
    assert all(label in title for title in asl_titles)


def test_default_titles_use_the_plain_asl_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_both(tmp_path)
    report = raw_data_eda.load_report(tmp_path / "interim/cleaning_report.csv")
    titles = capture_titles(monkeypatch)
    raw_data_eda.plot_class_distribution(report, tmp_path / "a.png")
    assert any("ASL baseline" in title for title in titles)
    assert not any("sample" in title for title in titles)


def test_parse_args_reads_asl_label() -> None:
    assert raw_data_eda.parse_args([]).asl_label == "ASL baseline"
    args = raw_data_eda.parse_args(["--asl-label", "ASL baseline (sample)"])
    assert args.asl_label == "ASL baseline (sample)"


def test_run_uses_the_asl_label_in_the_figures(tmp_path: Path) -> None:
    config = clean_both(tmp_path)
    add_figures_dir(config, tmp_path)
    args = ["--config", str(config), "--asl-label", "ASL baseline (sample, 100 per class)"]
    assert raw_data_eda.main(args) == 0
    assert (tmp_path / "figures" / "class_distribution.png").stat().st_size > 0
