import csv
from pathlib import Path

from tools.download_assembly101 import FIXED_VIEWS, hf_recording_patterns, recordings_from_annotations, split_videos


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["video", "verb_cls"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_v8_pattern_for_selected_recording():
    patterns = hf_recording_patterns(["rec_a"], "v8")
    assert patterns == [f"recordings/rec_a/{FIXED_VIEWS['v8']}"]


def test_all_recordings_v8_uses_wildcard():
    patterns = hf_recording_patterns(["all"], "v8")
    assert patterns == [f"recordings/*/{FIXED_VIEWS['v8']}"]


def test_recording_selection_filters_by_verb(tmp_path: Path):
    rows = [
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "screw"},
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "unscrew"},
    ]
    for split in ("train.csv", "validation.csv", "test.csv"):
        write_split(tmp_path / split, rows)

    selected = recordings_from_annotations(tmp_path, {"screw", "unscrew"})
    assert selected == ["rec_b", "rec_c"]


def test_split_videos():
    assert split_videos("all") == ["all"]
    assert split_videos("rec_a, rec_b") == ["rec_a", "rec_b"]
