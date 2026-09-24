"""Download extra WLASL videos for pretraining without touching RGB caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_data import (  # noqa: E402
    DEFAULT_RAW_DIR,
    download_samples_json,
    download_videos,
    filter_samples,
    load_samples_index,
    select_top_glosses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WLASL videos only")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    samples = load_samples_index(download_samples_json(raw_dir=args.raw_dir))
    class_names = select_top_glosses(samples, num_classes=args.num_classes)
    selected = filter_samples(samples, class_names)
    videos_root = Path(args.raw_dir) / "videos"
    missing = []
    for rec in selected:
        local = videos_root / Path(rec["filepath"]).name
        if not local.exists() or local.stat().st_size == 0:
            missing.append(rec)
    print(
        f"top-{args.num_classes}: {len(selected)} videos, "
        f"{len(selected) - len(missing)} local, {len(missing)} to download"
    )
    if not missing:
        print("Nothing to download.")
        return
    download_videos(missing, raw_dir=args.raw_dir, max_workers=args.max_workers)
    print(f"Done. videos dir now has {len(list(videos_root.glob('*.mp4')))} files")


if __name__ == "__main__":
    main()
