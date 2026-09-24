"""Extract pose+hand landmark sequences for WLASL top-K glosses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_data import (  # noqa: E402
    DEFAULT_NUM_CLASSES,
    DEFAULT_RAW_DIR,
    download_samples_json,
    download_videos,
    filter_samples,
    load_samples_index,
    select_top_glosses,
)
from modules.recognition.wlasl_landmarks import (  # noqa: E402
    DEFAULT_LANDMARK_DIR,
    DEFAULT_NUM_FRAMES,
    build_landmark_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WLASL landmark sequences")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--train-views", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    samples_json = Path(args.raw_dir) / "samples.json"
    if samples_json.exists():
        print(f"Using existing {samples_json}")
    else:
        samples_json = download_samples_json(raw_dir=args.raw_dir)
    samples = load_samples_index(samples_json)
    class_names = select_top_glosses(samples, num_classes=args.num_classes)
    selected = filter_samples(samples, class_names)
    print(f"Selected {len(class_names)} glosses, {len(selected)} videos")

    if args.skip_download:
        videos_root = Path(args.raw_dir) / "videos"
        records = []
        for rec in selected:
            local = videos_root / Path(rec["filepath"]).name
            if local.exists():
                item = dict(rec)
                item["video_path"] = str(local)
                records.append(item)
        print(f"Using {len(records)} local videos")
    else:
        records = download_videos(selected, raw_dir=args.raw_dir)

    build_landmark_cache(
        records,
        class_names,
        out_dir=args.out_dir,
        num_frames=args.num_frames,
        train_views=args.train_views,
    )


if __name__ == "__main__":
    main()
