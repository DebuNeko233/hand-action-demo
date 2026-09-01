from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "cvml-nus/assembly101"
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


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def split_videos(value: str) -> list[str]:
    value = value.strip()
    if not value or value == "all":
        return ["all"]
    return [x.strip() for x in value.split(",") if x.strip()]


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


def download_hf(root: Path, videos: list[str], views: str, token: str | None, annotations: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install -r requirements.txt") from exc

    allow_patterns: list[str] = []
    if annotations:
        allow_patterns.append("annotations/fine-grained-annotations/**")
    allow_patterns.extend(hf_recording_patterns(videos, views))

    print(f"Downloading official Assembly101 from Hugging Face: {HF_REPO_ID}")
    print(f"views={views} videos={','.join(videos)}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )


def ensure_official_downloader(tool_dir: Path, refresh: bool) -> None:
    if tool_dir.exists() and refresh:
        shutil.rmtree(tool_dir)
    if not tool_dir.exists():
        tool_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", OFFICIAL_DOWNLOADER_REPO, str(tool_dir)])
    else:
        run(["git", "pull", "--ff-only"], cwd=tool_dir)


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
    if len(videos) != 1:
        raise ValueError("The official Google Drive downloader accepts one recording name at a time or 'all'.")

    tool_dir = root / "_official_downloader"
    ensure_official_downloader(tool_dir, refresh_tools)

    if client_secrets:
        shutil.copy2(client_secrets, tool_dir / "client_secrets.json")
    if credentials:
        shutil.copy2(credentials, tool_dir / "credentials.json")

    if authenticate:
        run([sys.executable, "authenticate.py"], cwd=tool_dir)

    out_dir = root / "recordings"
    cmd = [
        sys.executable,
        "download.py",
        "--videos", videos[0],
        "--views", views,
        "--outDir", str(out_dir),
    ]
    if resume:
        # The upstream script declares --resume as type=bool, so pass an explicit value.
        cmd.extend(["--resume", "True"])
    run(cmd, cwd=tool_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Assembly101 for the MediaPipe action pipeline.")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "assembly101")
    ap.add_argument("--source", choices=["hf", "gdrive"], default="hf",
                    help="Official Hugging Face distribution (recommended) or the linked official Google Drive scripts.")
    ap.add_argument("--views", choices=VIEW_CHOICES, default="v8",
                    help="Default v8 downloads one fixed RGB view. Use 'fixed' for all 8 RGB views.")
    ap.add_argument("--videos", default="all",
                    help="'all' or a comma-separated recording-name list. GDrive mode supports one name or all.")
    ap.add_argument("--hf-token", default=None,
                    help="Hugging Face token. You can also set HF_TOKEN. The official dataset is gated.")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--client-secrets", type=Path, default=None,
                    help="GDrive mode: client_secrets.json with your Assembly101 Drive access.")
    ap.add_argument("--credentials", type=Path, default=None,
                    help="GDrive mode: optional pre-generated credentials.json for headless machines.")
    ap.add_argument("--authenticate", action="store_true",
                    help="GDrive mode: run the upstream authenticate.py before downloading.")
    ap.add_argument("--refresh-tools", action="store_true",
                    help="GDrive mode: re-clone the official assembly101-download-scripts repository.")
    args = ap.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    videos = split_videos(args.videos)

    if args.source == "hf":
        download_hf(root, videos, args.views, args.hf_token, not args.skip_annotations)
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
        if not args.skip_annotations:
            print("\nGoogle Drive mode downloads recordings through the official linked scripts.")
            print("For automatic fine-grained annotations, run this command once with --source hf")
            print("and the same --root, or place annotations under:")
            print(root / "annotations" / "fine-grained-annotations")

    print("\nAssembly101 root:", root)
    print("Recordings:", root / "recordings")
    print("Fine-grained annotations:", root / "annotations" / "fine-grained-annotations")


if __name__ == "__main__":
    main()
