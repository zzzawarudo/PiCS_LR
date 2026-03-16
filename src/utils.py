import cv2
import numpy as np


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def get_frame_at_index(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame


def compute_frame_indices(cap: cv2.VideoCapture, ref_sec: float, tgt_sec_from_end: float):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Не удалось определить количество кадров.")

    ref_idx = int(ref_sec * fps)
    tgt_idx = total_frames - 1 - int(tgt_sec_from_end * fps)

    ref_idx = clamp(ref_idx, 0, total_frames - 1)
    tgt_idx = clamp(tgt_idx, 0, total_frames - 1)

    return fps, total_frames, ref_idx, tgt_idx


def select_point_on_frame(frame: np.ndarray, win_name: str = "Select point"):

    chosen = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            chosen["pt"] = (int(x), int(y))

    vis = frame.copy()
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        temp = vis.copy()
        if chosen["pt"] is not None:
            cv2.circle(temp, chosen["pt"], 6, (0, 255, 0), 2)
            cv2.putText(temp, f"Selected: {chosen['pt']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(temp, "Click LBM to choose point. ENTER to confirm, ESC to close.",
                    (10, temp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, temp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(win_name)
            return None
        if key in (10, 13):  # ENTER
            if chosen["pt"] is not None:
                cv2.destroyWindow(win_name)
                return chosen["pt"]


def make_roi(center_xy, radius, w, h):
    cx, cy = center_xy
    x1 = clamp(cx - radius, 0, w - 1)
    y1 = clamp(cy - radius, 0, h - 1)
    x2 = clamp(cx + radius, 0, w - 1)
    y2 = clamp(cy + radius, 0, h - 1)
    return x1, y1, x2, y2