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
        description="Detect anime shots and build a PDF from in-memory JPG pages."
    )
    parser.add_argument("video", type=Path, help="Input MP4/video file.")
    parser.add_argument("output", type=Path, help="Output PDF path.")
    parser.add_argument(
        "--max_len",
        type=float,
        help="Debug limit in seconds: only detect scenes in the first N seconds.",
    )
    return parser.parse_args()


def load_img2pdf():
    try:
        import img2pdf
    except ModuleNotFoundError:
        print(
            "missing dependency: img2pdf\n"
            "install it with: python -m pip install img2pdf",
            file=sys.stderr,
        )
        return None
    return img2pdf


def pick_frame(start, end) -> int:
    fps = start.get_framerate()
    start_frame = start.get_frames()
    end_frame = end.get_frames()
    last_frame = max(start_frame, end_frame - 1)

    target_frame = start_frame + round(CAPTURE_OFFSET_SEC * fps)
    if target_frame >= end_frame - round(TAIL_GUARD_SEC * fps):
        target_frame = start_frame + ((last_frame - start_frame) // 2)
    return max(start_frame, min(target_frame, last_frame))


def open_capture(video: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    return capture


def read_frame(capture: cv2.VideoCapture, frame_number: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    return ok, frame


def encode_jpg(capture: cv2.VideoCapture, frame_number: int) -> tuple[bool, int, bytes | None]:
    last_candidate = frame_number
    for offset in range(RETRY_FRAMES + 1):
        candidate = frame_number + offset
        last_candidate = candidate
        ok, frame = read_frame(capture, candidate)
        if not ok or frame is None:
            continue
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if ok:
            return True, candidate, encoded.tobytes()
    return False, last_candidate, None


def write_pdf(output: Path, pages: list[bytes], img2pdf) -> None:
    with output.open("wb") as handle:
        img2pdf.convert(*pages, outputstream=handle)


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    img2pdf = load_img2pdf()
    if img2pdf is None:
        return 1

    if not video.is_file():
        print(f"missing video: {video}", file=sys.stderr)
        return 1
    if output.exists() and output.is_dir():
        print(f"output exists and is a folder: {output}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
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
    pages = []
    try:
        for i, (start, end) in enumerate(
            tqdm(scenes, desc="Encoding PDF pages"),
            start=1,
        ):
            frame_number = pick_frame(start, end)
            ok, actual_frame, page = encode_jpg(capture, frame_number)
            if ok and page is not None:
                pages.append(page)
                continue
            skipped.append((i, frame_number))
            print(
                f"skipping shot {i}: could not read frame {frame_number}"
                f" or nearby frames up to {actual_frame}",
                file=sys.stderr,
            )
    finally:
        capture.release()

    if not pages:
        print("no PDF pages could be encoded", file=sys.stderr)
        return 3

    try:
        write_pdf(output, pages, img2pdf)
    except Exception as exc:
        print(f"failed to write PDF: {exc}", file=sys.stderr)
        return 1

    if skipped:
        print(f"skipped {len(skipped)} shots", file=sys.stderr)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
