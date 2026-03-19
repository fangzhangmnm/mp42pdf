#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, detect
from tqdm import tqdm

MIN_SCENE_LEN = 12
ADAPTIVE_THRESHOLD = 3.0
MIN_CONTENT_VAL = 15.0
CAPTURE_OFFSET_SEC = 0.25
TAIL_GUARD_SEC = 0.10
JPEG_QUALITY = 95
RETRY_FRAMES = 3
CELL_WIDTH = 240
PAGE_GAP = 8
PAGE_BG = 255


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
    parser.add_argument("--n_cols", type=int, default=4, help="Images per row.")
    parser.add_argument("--n_rows", type=int, default=10, help="Rows per PDF page.")
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
    scene_len = max(1, end_frame - start_frame)

    start_buffer = round(CAPTURE_OFFSET_SEC * fps)
    end_buffer = round(TAIL_GUARD_SEC * fps)
    total_buffer = start_buffer + end_buffer

    if total_buffer > 0 and scene_len <= total_buffer:
        scale = scene_len / total_buffer
        start_buffer = round(start_buffer * scale)
        end_buffer = round(end_buffer * scale)

    safe_start = start_frame + start_buffer
    safe_end = end_frame - 1 - end_buffer
    if safe_start > safe_end:
        safe_start = safe_end = start_frame + ((last_frame - start_frame) // 2)

    return max(start_frame, min(safe_start, last_frame))


def open_capture(video: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    return capture


def read_frame(capture: cv2.VideoCapture, frame_number: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    return ok, frame


def read_selected_frame(
    capture: cv2.VideoCapture,
    frame_number: int,
) -> tuple[bool, int, np.ndarray | None]:
    last_candidate = frame_number
    for offset in range(RETRY_FRAMES + 1):
        candidate = frame_number + offset
        last_candidate = candidate
        ok, frame = read_frame(capture, candidate)
        if not ok or frame is None:
            continue
        return True, candidate, frame
    return False, last_candidate, None


def make_page(frame: np.ndarray, n_cols: int, n_rows: int) -> tuple[np.ndarray, int, int]:
    frame_height, frame_width = frame.shape[:2]
    cell_width = CELL_WIDTH
    cell_height = max(1, round(cell_width * frame_height / frame_width))
    page_width = (n_cols * cell_width) + ((n_cols + 1) * PAGE_GAP)
    page_height = (n_rows * cell_height) + ((n_rows + 1) * PAGE_GAP)
    page = np.full((page_height, page_width, 3), PAGE_BG, dtype=np.uint8)
    return page, cell_width, cell_height


def place_frame(
    page: np.ndarray,
    frame: np.ndarray,
    slot_index: int,
    n_cols: int,
    cell_width: int,
    cell_height: int,
) -> None:
    col = slot_index % n_cols
    row = slot_index // n_cols

    scale = min(cell_width / frame.shape[1], cell_height / frame.shape[0])
    width = max(1, round(frame.shape[1] * scale))
    height = max(1, round(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (width, height), interpolation=interpolation)

    cell_left = PAGE_GAP + col * (cell_width + PAGE_GAP)
    cell_top = PAGE_GAP + row * (cell_height + PAGE_GAP)
    x = cell_left + (cell_width - width) // 2
    y = cell_top + (cell_height - height) // 2
    page[y : y + height, x : x + width] = resized


def encode_page(page: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        page,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("failed to encode PDF page")
    return encoded.tobytes()


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
    if args.n_cols < 1 or args.n_rows < 1:
        print("n_cols and n_rows must be >= 1", file=sys.stderr)
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
    page = None
    cell_width = 0
    cell_height = 0
    slots_per_page = args.n_cols * args.n_rows
    slot_index = 0
    try:
        for i, (start, end) in enumerate(
            tqdm(scenes, desc="Building PDF"),
            start=1,
        ):
            frame_number = pick_frame(start, end)
            ok, actual_frame, frame = read_selected_frame(capture, frame_number)
            if ok and frame is not None:
                if page is None:
                    page, cell_width, cell_height = make_page(
                        frame,
                        args.n_cols,
                        args.n_rows,
                    )
                place_frame(
                    page,
                    frame,
                    slot_index,
                    args.n_cols,
                    cell_width,
                    cell_height,
                )
                slot_index += 1
                if slot_index == slots_per_page:
                    pages.append(encode_page(page))
                    page = None
                    slot_index = 0
                continue
            skipped.append((i, frame_number))
            print(
                f"skipping shot {i}: could not read frame {frame_number}"
                f" or nearby frames up to {actual_frame}",
                file=sys.stderr,
            )
    finally:
        capture.release()

    if page is not None and slot_index > 0:
        pages.append(encode_page(page))

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
