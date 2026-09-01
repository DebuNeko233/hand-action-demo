from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "cvml-nus/assembly101"
HF_CN_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_DOWNLOADER_REPO = "https://github.com/assembly-101/assembly101-download-scripts.git"

FIXED_VIEWS = {
    "v1": "C10095_rgb.mp4",
    "v2": "C10115_rgb.mp4",
    "v3": "C10118_rgb.mp4",
    "v4": "C10119_rgb.mp4",
    "v5": "C10379_rgb.mp4",
    "v6": "C10390_rgb.mp4",
    "v7": "C10395_rgb.mp4",
    "v8": "C10404_rgb.mp4",
}
EGO_PATTERNS = {
    "e1": ["HMC_84346135_mono10bit.mp4", "HMC_21176875_mono10bit.mp4"],
    "e2": ["HMC_84347414_mono10bit.mp4", "HMC_21176623_mono10bit.mp4"],
    "e3": ["HMC_84355350_mono10bit.mp4", "HMC_21110305_mono10bit.mp4"],
    "e4": ["HMC_84358933_mono10bit.mp4", "HMC_21179183_mono10bit.mp4"],
}
VIEW_CHOICES = ["all", "fixed", "egocentric", *FIXED_VIEWS, *EGO_PATTERNS]
REQUIRED_ANNOTATION_FILES = ("train.csv", "validation.csv", "test.csv")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def split_videos(value: str) -> list[str]:
    value = value.strip()
    if not value or value == "all":
        return ["all"]
    return split_csv_arg(value)


def configure_hf_endpoint(mirror: str) -> str:
    """Configure Hugging Face Hub endpoint before importing huggingface_hub."""
    if mirror == "cn":
        os.environ["HF_ENDPOINT"] = HF_CN_ENDPOINT
    elif mirror == "official":
        # Respect an explicitly supplied HF_ENDPOINT; otherwise use the official default.
        os.environ.pop("HF_ENDPOINT", None)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"Hugging Face endpoint: {endpoint}")
    return endpoint


def ensure_git_repo(url: str, target: Path, refresh: bool = False) -> None:
    if refresh and target.exists():
        shutil.rmtree(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", url, str(target)])
        return
    if not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a git checkout. Move it or choose another --root.")
    run(["git", "pull", "--ff-only"], cwd=target)


def hf_snapshot(root: Path, allow_patterns: list[str], token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install -r requirements.txt") from exc

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )


def find_annotation_dir(root: Path) -> Path | None:
    annotations_root = root / "annotations"
    if not annotations_root.exists():
        return None
    for train_csv in annotations_root.rglob("train.csv"):
        parent = train_csv.parent
        if all((parent / name).exists() for name in REQUIRED_ANNOTATION_FILES):
            return parent
    return None


def normalize_annotation_dir(root: Path, source_dir: Path) -> Path:
    target = root / "annotations" / "fine-grained-annotations"
    if source_dir.resolve() == target.resolve():
        return target
    target.mkdir(parents=True, exist_ok=True)
    for name in ("actions.csv", *REQUIRED_ANNOTATION_FILES, "head_actions.txt"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
    return target


def ensure_annotations(root: Path, token: str | None) -> Path:
    existing = find_annotation_dir(root)
    if existing is None:
        print("Downloading official Assembly101 annotations from Hugging Face...")
        print("The dataset is gated; accept the Assembly101 access terms before running this command.")
        hf_snapshot(root, ["annotations/**"], token)
        existing = find_annotation_dir(root)
    if existing is None:
        raise FileNotFoundError(
            "Could not locate Assembly101 train.csv/validation.csv/test.csv under the downloaded annotations tree."
        )
    return normalize_annotation_dir(root, existing)


def recording_verb_counts(annotation_dir: Path, split_file: str, labels: set[str] | None = None) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    path = annotation_dir / split_file
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            verb = row.get("verb_cls", "").strip().lower()
            if labels is not None and verb not in labels:
                continue
            video = row.get("video", "").replace("\\", "/")
            if "/" not in video:
                continue
            recording = video.split("/", 1)[0]
            counts[recording][verb] += 1
    return dict(counts)


def recordings_from_split(annotation_dir: Path, split_file: str, labels: set[str] | None = None) -> list[str]:
    return sorted(recording_verb_counts(annotation_dir, split_file, labels))


def recordings_from_annotations(annotation_dir: Path, labels: set[str] | None = None) -> list[str]:
    recordings: set[str] = set()
    for split_file in REQUIRED_ANNOTATION_FILES:
        recordings.update(recordings_from_split(annotation_dir, split_file, labels))
    return sorted(recordings)


def select_covering_recordings(
    annotation_dir: Path,
    split_file: str,
    labels: set[str] | None,
    max_recordings: int,
) -> tuple[list[str], set[str], dict[str, Counter[str]]]:
    counts = recording_verb_counts(annotation_dir, split_file, labels)
    if not counts:
        return [], set(), {}
    target_labels = set(labels) if labels is not None else {verb for c in counts.values() for verb in c}
    remaining = dict(counts)
    selected: list[str] = []
    covered: set[str] = set()

    while remaining and (max_recordings <= 0 or len(selected) < max_recordings):
        def score(item: tuple[str, Counter[str]]) -> tuple[int, int, int, str]:
            name, verb_counts = item
            verbs = set(verb_counts)
            new_coverage = len(verbs - covered)
            target_count = sum(verb_counts[v] for v in target_labels if v in verb_counts)
            balance = min((verb_counts[v] for v in target_labels if v in verb_counts), default=0)
            return new_coverage, len(verbs & target_labels), balance + target_count, name

        best_name, best_counts = max(remaining.items(), key=score)
        selected.append(best_name)
        covered.update(best_counts)
        del remaining[best_name]
        if covered >= target_labels:
            break

    return selected, covered & target_labels, counts


def recordings_per_split(annotation_dir: Path, labels: set[str] | None, max_per_split: int) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    result: dict[str, list[str]] = {}
    coverage: dict[str, set[str]] = {}
    for split_file in REQUIRED_ANNOTATION_FILES:
        split = Path(split_file).stem
        selected, covered, _ = select_covering_recordings(annotation_dir, split_file, labels, max_per_split)
        result[split] = selected
        coverage[split] = covered
    return result, coverage


def hf_recording_patterns(videos: list[str], views: str) -> list[str]:
    prefixes = ["recordings/*"] if videos == ["all"] else [f"recordings/{name}" for name in videos]

    if views == "all":
        names = ["*.mp4"]
    elif views == "fixed":
        names = list(FIXED_VIEWS.values())
    elif views == "egocentric":
        names = [name for names in EGO_PATTERNS.values() for name in names]
    elif views in FIXED_VIEWS:
        names = [FIXED_VIEWS[views]]
    else:
        names = EGO_PATTERNS[views]

    return [f"{prefix}/{name}" for prefix in prefixes for name in names]


def download_hf(root: Path, videos: list[str], views: str, token: str | None) -> None:
    allow_patterns = hf_recording_patterns(videos, views)
    print(f"Downloading official Assembly101 videos from Hugging Face: {HF_REPO_ID}")
    print(f"views={views} recordings={len(videos) if videos != ['all'] else 'all'}")
    print("The Hugging Face dataset is gated: accept its access terms before running this command.")
    hf_snapshot(root, allow_patterns, token)


def download_gdrive(
    root: Path,
    videos: list[str],
    views: str,
    client_secrets: Path | None,
    credentials: Path | None,
    authenticate: bool,
    resume: bool,
    refresh_tools: bool,
) -> None:
    tool_dir = root / "_official_downloader"
    ensure_git_repo(OFFICIAL_DOWNLOADER_REPO, tool_dir, refresh=refresh_tools)

    if client_secrets:
        shutil.copy2(client_secrets, tool_dir / "client_secrets.json")
    if credentials:
        shutil.copy2(credentials, tool_dir / "credentials.json")

    if authenticate:
        run([sys.executable, "authenticate.py"], cwd=tool_dir)

    for video in videos:
        cmd = [
            sys.executable,
            "download.py",
            "--videos", video,
            "--views", views,
            "--outDir", str(root / "recordings"),
        ]
        if resume:
            cmd.extend(["--resume", "True"])
        run(cmd, cwd=tool_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Assembly101 videos + official annotations for this project.")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "assembly101")
    ap.add_argument("--source", choices=["hf", "gdrive"], default="hf",
                    help="Hugging Face is recommended; gdrive wraps assembly101-download-scripts.")
    ap.add_argument("--mirror", choices=["official", "cn"], default="official",
                    help="Hugging Face endpoint. 'cn' uses https://hf-mirror.com.")
    ap.add_argument("--views", choices=VIEW_CHOICES, default="v8",
                    help="Default v8 downloads one fixed RGB view. Use 'fixed' for all 8 RGB views.")
    ap.add_argument("--videos", default="all",
                    help="'all' or comma-separated Assembly101 recording names.")
    ap.add_argument("--labels", default=None,
                    help="Optional comma-separated verb labels used to choose recordings, e.g. 'pick up,screw,unscrew'.")
    ap.add_argument("--max-recordings", type=int, default=0,
                    help="When --videos=all, choose up to N recordings from EACH official split, prioritizing target-verb coverage.")
    ap.add_argument("--hf-token", default=None,
                    help="Hugging Face token; HF_TOKEN environment variable is also supported.")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--client-secrets", type=Path, default=None,
                    help="GDrive mode: client_secrets.json with Assembly101 Drive access.")
    ap.add_argument("--credentials", type=Path, default=None,
                    help="GDrive mode: optional credentials.json for a remote/headless machine.")
    ap.add_argument("--authenticate", action="store_true",
                    help="GDrive mode: run upstream authenticate.py before downloading.")
    ap.add_argument("--refresh-tools", action="store_true",
                    help="Re-clone the official Google Drive downloader helper repository.")
    args = ap.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        configure_hf_endpoint(args.mirror)

    annotation_dir: Path | None = None
    if not args.skip_annotations or args.max_recordings > 0 or args.labels:
        annotation_dir = ensure_annotations(root, args.hf_token)

    videos = split_videos(args.videos)
    if videos == ["all"] and (args.max_recordings > 0 or args.labels):
        if annotation_dir is None:
            annotation_dir = ensure_annotations(root, args.hf_token)
        label_filter = {x.lower() for x in split_csv_arg(args.labels)} or None
        if args.max_recordings > 0:
            by_split, coverage = recordings_per_split(annotation_dir, label_filter, args.max_recordings)
            videos = sorted({name for names in by_split.values() for name in names})
            print("Selected recordings per official split:")
            for split, names in by_split.items():
                covered = ", ".join(sorted(coverage[split])) or "none"
                print(f"  {split}: {len(names)} recording(s), covered verbs: {covered}")
                if label_filter is not None:
                    missing = sorted(label_filter - coverage[split])
                    if missing:
                        print(f"    missing with current --max-recordings: {', '.join(missing)}")
        else:
            videos = recordings_from_annotations(annotation_dir, label_filter)
        if not videos:
            raise RuntimeError("No Assembly101 recordings matched the requested verb labels.")
        print(f"Total unique recordings selected: {len(videos)}")

    if args.source == "hf":
        download_hf(root, videos, args.views, args.hf_token)
    else:
        download_gdrive(
            root=root,
            videos=videos,
            views=args.views,
            client_secrets=args.client_secrets,
            credentials=args.credentials,
            authenticate=args.authenticate,
            resume=args.resume,
            refresh_tools=args.refresh_tools,
        )

    print("\nAssembly101 root:", root)
    print("Recordings:", root / "recordings")
    if annotation_dir is not None:
        print("Fine-grained annotations:", annotation_dir)
    print("\nNext step:")
    print(
        f"  {sys.executable} tools/prepare_assembly101.py --video-root \"{root / 'recordings'}\" "
        f"--annotation-dir \"{root / 'annotations' / 'fine-grained-annotations'}\""
    )


if __name__ == "__main__":
    main()
