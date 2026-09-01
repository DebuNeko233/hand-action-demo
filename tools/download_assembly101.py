from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "cvml-nus/assembly101"
OFFICIAL_DOWNLOADER_REPO = "https://github.com/assembly-101/assembly101-download-scripts.git"
OFFICIAL_ANNOTATIONS_REPO = "https://github.com/assembly-101/assembly101-annotations.git"

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


def ensure_annotations(root: Path, refresh: bool = False) -> Path:
    annotation_repo = root / "annotations"
    ensure_git_repo(OFFICIAL_ANNOTATIONS_REPO, annotation_repo, refresh=refresh)
    annotation_dir = annotation_repo / "fine-grained-annotations"
    required = [annotation_dir / x for x in ("train.csv", "validation.csv", "test.csv")]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Official Assembly101 annotation checkout is incomplete: " + ", ".join(missing))
    return annotation_dir


def recordings_from_annotations(annotation_dir: Path, labels: set[str] | None = None) -> list[str]:
    recordings: set[str] = set()
    for split_file in ("train.csv", "validation.csv", "test.csv"):
        path = annotation_dir / split_file
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if labels is not None and row.get("verb_cls", "").strip().lower() not in labels:
                    continue
                video = row.get("video", "").replace("\\", "/")
                if "/" in video:
                    recordings.add(video.split("/", 1)[0])
    return sorted(recordings)


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
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install -r requirements.txt") from exc

    allow_patterns = hf_recording_patterns(videos, views)
    print(f"Downloading official Assembly101 from Hugging Face: {HF_REPO_ID}")
    print(f"views={views} recordings={len(videos) if videos != ['all'] else 'all'}")
    print("The Hugging Face dataset is gated: accept its access terms before running this command.")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )


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

    # The upstream CLI accepts only one recording name at a time (or 'all'),
    # so run it repeatedly when our wrapper receives a selected list.
    for video in videos:
        cmd = [
            sys.executable,
            "download.py",
            "--videos", video,
            "--views", views,
            "--outDir", str(root / "recordings"),
        ]
        if resume:
            # Upstream currently declares --resume as type=bool.
            cmd.extend(["--resume", "True"])
        run(cmd, cwd=tool_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Assembly101 videos + official annotations for this project.")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "assembly101")
    ap.add_argument("--source", choices=["hf", "gdrive"], default="hf",
                    help="Hugging Face is recommended; gdrive wraps assembly101-download-scripts exactly.")
    ap.add_argument("--views", choices=VIEW_CHOICES, default="v8",
                    help="Default v8 downloads one fixed RGB view. Use 'fixed' for all 8 RGB views.")
    ap.add_argument("--videos", default="all",
                    help="'all' or comma-separated Assembly101 recording names.")
    ap.add_argument("--labels", default=None,
                    help="Optional comma-separated verb labels used only to choose recordings, e.g. 'pick up,screw,unscrew'.")
    ap.add_argument("--max-recordings", type=int, default=0,
                    help="When --videos=all, restrict download to the first N recordings matched by --labels. Useful for smoke tests.")
    ap.add_argument("--hf-token", default=None,
                    help="Hugging Face token; HF_TOKEN environment variable is also supported.")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--client-secrets", type=Path, default=None,
                    help="GDrive mode: client_secrets.json with Assembly101 Drive access.")
    ap.add_argument("--credentials", type=Path, default=None,
                    help="GDrive mode: optional credentials.json for a remote/headless machine.")
    ap.add_argument("--authenticate", action="store_true",
                    help="GDrive mode: run the upstream authenticate.py before downloading.")
    ap.add_argument("--refresh-tools", action="store_true",
                    help="Re-clone official download/annotation helper repositories.")
    args = ap.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    annotation_dir: Path | None = None
    if not args.skip_annotations or args.max_recordings > 0 or args.labels:
        annotation_dir = ensure_annotations(root, refresh=args.refresh_tools)

    videos = split_videos(args.videos)
    if videos == ["all"] and (args.max_recordings > 0 or args.labels):
        if annotation_dir is None:
            annotation_dir = ensure_annotations(root, refresh=False)
        label_filter = {x.lower() for x in split_csv_arg(args.labels)} or None
        videos = recordings_from_annotations(annotation_dir, label_filter)
        if args.max_recordings > 0:
            videos = videos[:args.max_recordings]
        if not videos:
            raise RuntimeError("No Assembly101 recordings matched the requested verb labels.")
        print(f"Selected {len(videos)} recordings from official annotations.")

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
    print(f"  {sys.executable} tools/prepare_assembly101.py --video-root \"{root / 'recordings'}\" --annotation-dir \"{root / 'annotations' / 'fine-grained-annotations'}\"")


if __name__ == "__main__":
    main()
