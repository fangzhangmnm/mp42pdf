#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
from typing import Callable

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, open_video
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm

MIN_SCENE_LEN = 12
ADAPTIVE_THRESHOLD = 3.0
MIN_CONTENT_VAL = 15.0
CAPTURE_OFFSET_SEC = 0.25
TAIL_GUARD_SEC = 0.10
JPEG_QUALITY = 95
RETRY_FRAMES = 3
DEFAULT_N_COLS = 4
DEFAULT_N_ROWS = 10
CELL_WIDTH = 240
PAGE_GAP = 8
PAGE_BG = 255


class AppError(Exception):
    pass


@dataclass
class RunResult:
    output: Path
    scene_count: int
    page_count: int
    skipped_count: int


ProgressCallback = Callable[[str, int | None, int | None], None]


def load_img2pdf():
    try:
        import img2pdf
    except ModuleNotFoundError as exc:
        raise AppError(
            "missing dependency: img2pdf\n"
            "install it with: python -m pip install img2pdf"
        ) from exc
    return img2pdf


def default_output_path(video: Path) -> Path:
    base = video.with_suffix(".pdf")
    if not base.exists():
        return base
    for index in range(2, 10_000):
        candidate = base.with_name(f"{base.stem} ({index}){base.suffix}")
        if not candidate.exists():
            return candidate
    raise AppError(f"could not find a free output name near {base}")


def normalize_output_path(video: Path, output: Path | None) -> Path:
    if output is None:
        return default_output_path(video)
    output = output.expanduser()
    if output.suffix == "":
        output = output.with_suffix(".pdf")
    return output.resolve()


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
        raise AppError(f"cannot open video: {video}")
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
        raise AppError("failed to encode PDF page")
    return encoded.tobytes()


def write_pdf(output: Path, pages: list[bytes], img2pdf) -> None:
    with output.open("wb") as handle:
        img2pdf.convert(*pages, outputstream=handle)


def detect_scene_list(
    video_path: Path,
    *,
    max_len: float | None = None,
    show_progress: bool = False,
    progress: ProgressCallback | None = None,
):
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=ADAPTIVE_THRESHOLD,
            min_scene_len=MIN_SCENE_LEN,
            min_content_val=MIN_CONTENT_VAL,
        )
    )

    total_frames = video.duration.get_frames() if video.duration is not None else None
    end_time = None
    if max_len is not None:
        end_time = FrameTimecode(timecode=float(max_len), fps=video.frame_rate)
        end_frames = end_time.get_frames()
        if total_frames is None:
            total_frames = end_frames
        else:
            total_frames = min(total_frames, end_frames)

    monitor_done = threading.Event()
    monitor_thread = None
    if progress and total_frames is not None:
        progress("Detecting scenes...", 0, total_frames)

        def monitor() -> None:
            last_reported = -1
            step = max(1, total_frames // 200)
            while not monitor_done.wait(0.05):
                current = min(video.position.frame_num, total_frames)
                if current >= total_frames or current - last_reported >= step:
                    last_reported = current
                    progress("Detecting scenes...", current, total_frames)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    try:
        manager.detect_scenes(
            video=video,
            end_time=end_time,
            show_progress=show_progress,
        )
    finally:
        if monitor_thread is not None:
            monitor_done.set()
            monitor_thread.join()
    scenes = manager.get_scene_list(start_in_scene=True)
    if progress and total_frames is not None:
        progress("Detecting scenes...", total_frames, total_frames)
    return scenes


def process_video(
    video: Path,
    output: Path | None = None,
    *,
    max_len: float | None = None,
    n_cols: int = DEFAULT_N_COLS,
    n_rows: int = DEFAULT_N_ROWS,
    show_detect_progress: bool = False,
    show_build_progress: bool = False,
    progress: ProgressCallback | None = None,
) -> RunResult:
    img2pdf = load_img2pdf()
    video = video.expanduser().resolve()
    output = normalize_output_path(video, output)

    if n_cols < 1 or n_rows < 1:
        raise AppError("n_cols and n_rows must be >= 1")
    if not video.is_file():
        raise AppError(f"missing video: {video}")
    if output.exists() and output.is_dir():
        raise AppError(f"output exists and is a folder: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    scenes = detect_scene_list(
        video,
        max_len=max_len,
        show_progress=show_detect_progress,
        progress=progress,
    )
    if not scenes:
        raise AppError("no scenes found")

    capture = open_capture(video)
    skipped = 0
    pages: list[bytes] = []
    page = None
    cell_width = 0
    cell_height = 0
    slots_per_page = n_cols * n_rows
    slot_index = 0
    total = len(scenes)
    iterator = tqdm(scenes, desc="Building PDF") if show_build_progress else scenes

    try:
        if progress:
            progress("Building PDF...", 0, total)
        for i, (start, end) in enumerate(iterator, start=1):
            frame_number = pick_frame(start, end)
            ok, actual_frame, frame = read_selected_frame(capture, frame_number)
            if ok and frame is not None:
                if page is None:
                    page, cell_width, cell_height = make_page(frame, n_cols, n_rows)
                place_frame(page, frame, slot_index, n_cols, cell_width, cell_height)
                slot_index += 1
                if slot_index == slots_per_page:
                    pages.append(encode_page(page))
                    page = None
                    slot_index = 0
            else:
                skipped += 1
                if show_build_progress:
                    tqdm.write(
                        f"skipping shot {i}: could not read frame {frame_number}"
                        f" or nearby frames up to {actual_frame}"
                    )
            if progress:
                progress("Building PDF...", i, total)
    finally:
        capture.release()

    if page is not None and slot_index > 0:
        pages.append(encode_page(page))
    if not pages:
        raise AppError("no PDF pages could be encoded")

    if progress:
        progress("Writing PDF...", None, None)
    write_pdf(output, pages, img2pdf)
    if progress:
        progress("Done", total, total)

    return RunResult(
        output=output,
        scene_count=total,
        page_count=len(pages),
        skipped_count=skipped,
    )


##########


def launch_ui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("mp42pdf")
    root.resizable(False, False)

    video_var = tk.StringVar()
    output_var = tk.StringVar()
    max_len_var = tk.StringVar()
    n_cols_var = tk.StringVar(value=str(DEFAULT_N_COLS))
    n_rows_var = tk.StringVar(value=str(DEFAULT_N_ROWS))
    status_var = tk.StringVar(value="Choose a video to begin.")
    event_queue: queue.Queue = queue.Queue()

    def auto_output(video_path: str) -> None:
        if not video_path:
            return
        try:
            output_var.set(str(default_output_path(Path(video_path).expanduser().resolve())))
        except Exception:
            pass

    def pick_video() -> None:
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi"), ("All files", "*.*")],
        )
        if path:
            video_var.set(path)
            auto_output(path)

    def pick_output() -> None:
        initial = output_var.get()
        initial_path = Path(initial) if initial else None
        path = filedialog.asksaveasfilename(
            title="Save PDF As",
            defaultextension=".pdf",
            initialdir=str(initial_path.parent) if initial_path else "",
            initialfile=initial_path.name if initial_path else "",
            filetypes=[("PDF", "*.pdf")],
        )
        if path:
            output_var.set(path)

    def set_running(running: bool) -> None:
        state = "disabled" if running else "normal"
        for widget in controls:
            widget.configure(state=state)
        progress.configure(mode="indeterminate" if running else "determinate")
        if running:
            progress.start(10)
        else:
            progress.stop()

    def run_worker() -> None:
        try:
            max_len = float(max_len_var.get()) if max_len_var.get().strip() else None
            result = process_video(
                Path(video_var.get()),
                Path(output_var.get()) if output_var.get().strip() else None,
                max_len=max_len,
                n_cols=int(n_cols_var.get()),
                n_rows=int(n_rows_var.get()),
                progress=lambda phase, current, total: event_queue.put(
                    ("progress", phase, current, total)
                ),
            )
        except Exception as exc:
            event_queue.put(("error", str(exc)))
        else:
            event_queue.put(("done", result))

    def start() -> None:
        if not video_var.get().strip():
            messagebox.showerror("mp42pdf", "Choose a video first.")
            return
        if not output_var.get().strip():
            auto_output(video_var.get())
        set_running(True)
        status_var.set("Starting...")
        threading.Thread(target=run_worker, daemon=True).start()

    def poll_queue() -> None:
        try:
            while True:
                kind, *payload = event_queue.get_nowait()
                if kind == "progress":
                    phase, current, total = payload
                    if current is not None and total:
                        progress.configure(mode="determinate", maximum=total, value=current)
                        status_var.set(f"{phase} {current}/{total}")
                    else:
                        progress.configure(mode="indeterminate")
                        status_var.set(phase)
                elif kind == "error":
                    set_running(False)
                    status_var.set("Failed.")
                    messagebox.showerror("mp42pdf", payload[0])
                elif kind == "done":
                    result = payload[0]
                    set_running(False)
                    progress.configure(mode="determinate", maximum=max(1, result.page_count), value=result.page_count)
                    status_var.set(
                        f"Done. {result.scene_count} scenes, {result.page_count} pages."
                    )
                    messagebox.showinfo("mp42pdf", f"Saved PDF to:\n{result.output}")
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    frame = ttk.Frame(root, padding=12)
    frame.grid()

    ttk.Label(frame, text="Video").grid(row=0, column=0, sticky="w")
    video_entry = ttk.Entry(frame, width=42, textvariable=video_var)
    video_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
    video_button = ttk.Button(frame, text="Browse...", command=pick_video)
    video_button.grid(row=1, column=1, sticky="ew")

    ttk.Label(frame, text="Output PDF").grid(row=2, column=0, sticky="w", pady=(10, 0))
    output_entry = ttk.Entry(frame, width=42, textvariable=output_var)
    output_entry.grid(row=3, column=0, sticky="ew", padx=(0, 8))
    output_button = ttk.Button(frame, text="Save As...", command=pick_output)
    output_button.grid(row=3, column=1, sticky="ew")

    ttk.Label(frame, text="max_len").grid(row=4, column=0, sticky="w", pady=(10, 0))
    max_len_entry = ttk.Entry(frame, width=12, textvariable=max_len_var)
    max_len_entry.grid(row=5, column=0, sticky="w")

    ttk.Label(frame, text="n_cols").grid(row=4, column=0, sticky="w", padx=(120, 0), pady=(10, 0))
    n_cols_entry = ttk.Entry(frame, width=8, textvariable=n_cols_var)
    n_cols_entry.grid(row=5, column=0, sticky="w", padx=(120, 0))

    ttk.Label(frame, text="n_rows").grid(row=4, column=0, sticky="w", padx=(210, 0), pady=(10, 0))
    n_rows_entry = ttk.Entry(frame, width=8, textvariable=n_rows_var)
    n_rows_entry.grid(row=5, column=0, sticky="w", padx=(210, 0))

    progress = ttk.Progressbar(frame, length=360, mode="determinate")
    progress.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 6))
    ttk.Label(frame, textvariable=status_var).grid(row=7, column=0, columnspan=2, sticky="w")

    start_button = ttk.Button(frame, text="Generate storyboard PDF from video", command=start)
    start_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    controls = [
        video_entry,
        video_button,
        output_entry,
        output_button,
        max_len_entry,
        n_cols_entry,
        n_rows_entry,
        start_button,
    ]

    frame.columnconfigure(0, weight=1)
    root.after(100, poll_queue)
    root.mainloop()
    return 0


##########


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect scenes in a video file, and build a storyboard PDF."
    )
    parser.add_argument("video", type=Path, help="Input MP4/video file.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output PDF path. Defaults to <video>.pdf next to the source file.",
    )
    parser.add_argument(
        "--max_len",
        type=float,
        help="Maximum length of video to process, in seconds. By default, the entire video is processed.",
    )
    parser.add_argument("--n_cols", type=int, default=DEFAULT_N_COLS, help="Images per row.")
    parser.add_argument("--n_rows", type=int, default=DEFAULT_N_ROWS, help="Rows per PDF page.")
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> int:
    try:
        result = process_video(
            args.video,
            args.output,
            max_len=args.max_len,
            n_cols=args.n_cols,
            n_rows=args.n_rows,
            show_detect_progress=True,
            show_build_progress=True,
        )
    except AppError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if result.skipped_count:
        print(f"skipped {result.skipped_count} shots", file=sys.stderr)
    print(result.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return launch_ui()
    return run_cli(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
