#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
from scenedetect import AdaptiveDetector, detect
from tqdm import tqdm

MIN_SCENE_LEN = 12
ADAPTIVE_THRESHOLD = 3.0
MIN_CONTENT_VAL = 15.0
CAPTURE_OFFSET_SEC = 0.25
TAIL_GUARD_SEC = 0.10
JPEG_QUALITY = 95
RETRY_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect anime shots and save one JPG per scene."
    )
    parser.add_argument("video", type=Path, help="Input MP4/video file.")
    parser.add_argument(
        "output",
        type=Path,
        help="Output PDF name for now; its stem is used as the JPG folder name.",
    )
    parser.add_argument(
        "--max-len",
        type=float,
        help="Debug limit in seconds: only detect scenes in the first N seconds.",
    )
    return parser.parse_args()


def output_dir(target: Path) -> Path:
    return target.with_suffix("") if target.suffix else target


def stamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}-{m:02d}-{s:02d}-{ms:03d}"


def pick_frame(start, end) -> tuple[int, float]:
    fps = start.get_framerate()
    start_frame = start.get_frames()
    end_frame = end.get_frames()
    last_frame = max(start_frame, end_frame - 1)

    target_frame = start_frame + round(CAPTURE_OFFSET_SEC * fps)
    if target_frame >= end_frame - round(TAIL_GUARD_SEC * fps):
        target_frame = start_frame + ((last_frame - start_frame) // 2)
    target_frame = max(start_frame, min(target_frame, last_frame))
    return target_frame, target_frame / fps


def open_capture(video: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    return capture


def read_frame(capture: cv2.VideoCapture, frame_number: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    return ok, frame


def save_jpg(capture: cv2.VideoCapture, frame_number: int, jpg: Path) -> tuple[bool, int]:
    last_candidate = frame_number
    for offset in range(RETRY_FRAMES + 1):
        candidate = frame_number + offset
        last_candidate = candidate
        ok, frame = read_frame(capture, candidate)
        if not ok or frame is None:
            continue
        written = cv2.imwrite(
            str(jpg),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if written:
            return True, candidate
    return False, last_candidate


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    out = output_dir(args.output.expanduser().resolve())

    if not video.is_file():
        print(f"missing video: {video}", file=sys.stderr)
        return 1
    if out.exists() and not out.is_dir():
        print(f"output exists and is not a folder: {out}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    scenes = detect(
        str(video),
        AdaptiveDetector(
            adaptive_threshold=ADAPTIVE_THRESHOLD,
            min_scene_len=MIN_SCENE_LEN,
            min_content_val=MIN_CONTENT_VAL,
        ),
        show_progress=True,
        start_in_scene=True,
        end_time=args.max_len,
    )

    if not scenes:
        print("no scenes found", file=sys.stderr)
        return 2

    try:
        capture = open_capture(video)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    skipped = []
    for i, (start, end) in enumerate(tqdm(scenes, desc="Saving JPGs"), start=1):
        frame_number, seconds = pick_frame(start, end)
        ok, actual_frame = save_jpg(
            capture,
            frame_number,
            out / f"{i:04d}_{stamp(seconds)}.jpg",
        )
        if not ok:
            skipped.append((i, frame_number))
            print(
                f"skipping shot {i}: could not read frame {frame_number}"
                f" or nearby frames up to {actual_frame}",
                file=sys.stderr,
            )

    capture.release()

    if skipped:
        print(f"skipped {len(skipped)} shots", file=sys.stderr)

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
