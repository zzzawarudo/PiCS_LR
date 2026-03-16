import os
import csv
import math
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectors import available_methods, detect_in_roi, match_descriptors


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def get_frame_at_index(cap: cv2.VideoCapture, frame_idx: int):
    """
    Надёжное чтение кадра по индексу.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return False, None
        if i == frame_idx:
            return True, frame
        i += 1


def select_point_on_frame(frame: np.ndarray, win_name: str = "Select point"):
    """
    ЛКМ = выбрать точку, Enter = подтвердить, ESC = отмена.
    """
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
            cv2.putText(
                temp,
                f"Selected: {chosen['pt']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            temp,
            "LMB click to select point. ENTER = confirm, ESC = cancel.",
            (10, temp.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

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


def centroid(points: np.ndarray):
    """
    points: shape (N, 2)
    """
    if points is None or len(points) == 0:
        return None
    return points.mean(axis=0)


def mean_dist_to_centroid(points: np.ndarray):
    """
    Средняя дистанция точек до их центроида.
    """
    if points is None or len(points) == 0:
        return None
    c = centroid(points)
    d = np.linalg.norm(points - c, axis=1)
    return float(d.mean())


def std_dist_to_centroid(points: np.ndarray):
    """
    Стандартное отклонение дистанций до центроида.
    """
    if points is None or len(points) == 0:
        return None
    c = centroid(points)
    d = np.linalg.norm(points - c, axis=1)
    return float(d.std())


def estimate_homography(matches, kps_ref, kps_cur):
    """
    Считает гомографию ref -> current по good matches.
    Возвращает H, mask, ref_pts, cur_pts.
    """
    if len(matches) < 4:
        return None, None, None, None

    ref_pts = np.float32([kps_ref[m.queryIdx].pt for m in matches])
    cur_pts = np.float32([kps_cur[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, None, None, None

    mask = mask.ravel().astype(bool)
    return H, mask, ref_pts, cur_pts


def warp_points(points: np.ndarray, H: np.ndarray):
    """
    Переносит точки через гомографию.
    points shape = (N, 2)
    """
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)


def euclidean(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def save_method_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_sec",
            "kps_ref",
            "kps_cur",
            "good_matches",
            "inliers",
            "inlier_ratio",
            "centroid_ref_x",
            "centroid_ref_y",
            "centroid_warped_x",
            "centroid_warped_y",
            "centroid_shift_px",
            "mean_dist_ref",
            "mean_dist_warped",
            "shape_drift_abs",
            "shape_drift_rel",
            "reprojection_error_px",
            "status",
        ])
        writer.writerows(rows)


def plot_metric(method_to_rows, metric_name, out_path, ylabel):
    plt.figure(figsize=(10, 6))
    for method_name, rows in method_to_rows.items():
        xs = []
        ys = []
        for r in rows:
            # frame_idx=0, time_sec=1, ... metric depends
            metric_map = {
                "centroid_shift_px": 11,
                "shape_drift_abs": 14,
                "shape_drift_rel": 15,
                "reprojection_error_px": 16,
                "inliers": 5,
                "inlier_ratio": 6,
                "kps_cur": 3,
                "good_matches": 4,
            }
            col = metric_map[metric_name]

            val = r[col]
            if val == "" or val is None:
                continue
            xs.append(r[0])  # frame_idx
            ys.append(float(val))

        if xs:
            plt.plot(xs, ys, marker="o", markersize=3, label=method_name)

    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(metric_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_preview(frame, roi, pts=None, title="preview"):
    vis = frame.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if pts is not None:
        for p in pts:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)

    cv2.imshow(title, vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/input.mp4", help="Path to input video")
    parser.add_argument("--roi_radius", type=int, default=120, help="ROI radius in pixels")
    parser.add_argument("--ref_sec", type=float, default=0.5, help="Reference frame time from start")
    parser.add_argument("--step", type=int, default=5, help="Process every N-th frame")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold")
    parser.add_argument("--out_dir", type=str, default="outputs/stability", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Show preview windows")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть видео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Не удалось определить количество кадров.")

    ref_idx = int(args.ref_sec * fps)
    ref_idx = clamp(ref_idx, 0, total_frames - 1)

    ok, ref_frame = get_frame_at_index(cap, ref_idx)
    if not ok:
        raise RuntimeError("Не удалось считать reference frame.")

    pt = select_point_on_frame(ref_frame, "Select ROI center")
    if pt is None:
        print("Точка не выбрана. Выход.")
        return

    h, w = ref_frame.shape[:2]
    roi = make_roi(pt, args.roi_radius, w, h)

    # Сохраняем ref с ROI
    ref_vis = ref_frame.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(ref_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(ref_vis, pt, 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(args.out_dir, "reference_with_roi.png"), ref_vis)

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    methods = available_methods()
    if not methods:
        raise RuntimeError("Не удалось создать ни один дескриптор.")

    # Для каждого метода заранее считаем reference keypoints/descriptors
    method_data = {}
    for method in methods:
        kps_ref, desc_ref = detect_in_roi(method, ref_gray, roi)
        method_data[method.name] = {
            "method": method,
            "kps_ref": kps_ref,
            "desc_ref": desc_ref,
            "rows": [],
        }

    # Идём по кадрам
    for frame_idx in range(ref_idx + args.step, total_frames, args.step):
        ok, cur_frame = get_frame_at_index(cap, frame_idx)
        if not ok:
            continue

        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        for method_name, data in method_data.items():
            method = data["method"]
            kps_ref = data["kps_ref"]
            desc_ref = data["desc_ref"]

            kps_cur, desc_cur = detect_in_roi(method, cur_gray, roi)
            matches = match_descriptors(method, desc_ref, desc_cur, ratio=args.ratio)

            H, inlier_mask, ref_pts, cur_pts = estimate_homography(matches, kps_ref, kps_cur)

            if H is None:
                data["rows"].append([
                    frame_idx,
                    frame_idx / fps,
                    len(kps_ref),
                    len(kps_cur),
                    len(matches),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "homography_failed",
                ])
                continue

            inlier_ref = ref_pts[inlier_mask]
            inlier_cur = cur_pts[inlier_mask]

            if len(inlier_ref) < 4:
                data["rows"].append([
                    frame_idx,
                    frame_idx / fps,
                    len(kps_ref),
                    len(kps_cur),
                    len(matches),
                    len(inlier_ref),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "not_enough_inliers",
                ])
                continue

            # Переносим текущие inlier-точки обратно в reference-плоскость
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                data["rows"].append([
                    frame_idx,
                    frame_idx / fps,
                    len(kps_ref),
                    len(kps_cur),
                    len(matches),
                    int(inlier_mask.sum()) if inlier_mask is not None else "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "singular_homography",
                ])
                continue

            cur_warped_to_ref = warp_points(inlier_cur, H_inv)

            # Центроиды
            c_ref = centroid(inlier_ref)
            c_warped = centroid(cur_warped_to_ref)
            centroid_shift_px = euclidean(c_ref, c_warped)

            # Геометрия набора точек:
            # сравниваем среднее расстояние до центроида в ref и после обратного варпа
            mean_dist_ref = mean_dist_to_centroid(inlier_ref)
            mean_dist_warped = mean_dist_to_centroid(cur_warped_to_ref)

            shape_drift_abs = abs(mean_dist_warped - mean_dist_ref)
            shape_drift_rel = shape_drift_abs / mean_dist_ref if mean_dist_ref and mean_dist_ref > 1e-8 else None

            # Средняя ошибка после обратного варпа:
            # насколько точки "вернулись" туда, где были в reference
            reproj_error = np.linalg.norm(cur_warped_to_ref - inlier_ref, axis=1).mean()

            inliers_count = int(inlier_mask.sum())
            inlier_ratio = inliers_count / len(matches) if len(matches) > 0 else None

            data["rows"].append([
                frame_idx,
                frame_idx / fps,
                len(kps_ref),
                len(kps_cur),
                len(matches),
                inliers_count,
                f"{inlier_ratio:.6f}" if inlier_ratio is not None else "",
                float(c_ref[0]),
                float(c_ref[1]),
                float(c_warped[0]),
                float(c_warped[1]),
                float(centroid_shift_px),
                float(mean_dist_ref) if mean_dist_ref is not None else "",
                float(mean_dist_warped) if mean_dist_warped is not None else "",
                float(shape_drift_abs) if shape_drift_abs is not None else "",
                float(shape_drift_rel) if shape_drift_rel is not None else "",
                float(reproj_error),
                "ok",
            ])

            if args.preview:
                draw_preview(cur_frame, roi, pts=inlier_cur, title=f"{method_name} inliers")
                cv2.waitKey(1)

        print(f"Processed frame {frame_idx}/{total_frames - 1}")

    cap.release()
    cv2.destroyAllWindows()

    # Сохраняем CSV
    for method_name, data in method_data.items():
        csv_path = os.path.join(args.out_dir, f"stability_{method_name}.csv")
        save_method_csv(data["rows"], csv_path)

    # Графики
    method_to_rows = {k: v["rows"] for k, v in method_data.items()}

    plot_metric(
        method_to_rows,
        "centroid_shift_px",
        os.path.join(args.out_dir, "plot_centroid_shift.png"),
        "Centroid shift, px",
    )
    plot_metric(
        method_to_rows,
        "shape_drift_abs",
        os.path.join(args.out_dir, "plot_shape_drift_abs.png"),
        "Shape drift abs, px",
    )
    plot_metric(
        method_to_rows,
        "shape_drift_rel",
        os.path.join(args.out_dir, "plot_shape_drift_rel.png"),
        "Shape drift relative",
    )
    plot_metric(
        method_to_rows,
        "reprojection_error_px",
        os.path.join(args.out_dir, "plot_reprojection_error.png"),
        "Reprojection error, px",
    )
    plot_metric(
        method_to_rows,
        "inliers",
        os.path.join(args.out_dir, "plot_inliers.png"),
        "Inliers",
    )
    plot_metric(
        method_to_rows,
        "inlier_ratio",
        os.path.join(args.out_dir, "plot_inlier_ratio.png"),
        "Inlier ratio",
    )
    plot_metric(
        method_to_rows,
        "kps_cur",
        os.path.join(args.out_dir, "plot_kps_cur.png"),
        "Current keypoints",
    )
    plot_metric(
        method_to_rows,
        "good_matches",
        os.path.join(args.out_dir, "plot_good_matches.png"),
        "Good matches",
    )

    print("\nГотово.")
    print(f"Результаты сохранены в: {args.out_dir}")
    print("CSV по методам:")
    for method_name in method_data.keys():
        print(f"- stability_{method_name}.csv")
    print("Графики:")
    print("- plot_centroid_shift.png")
    print("- plot_shape_drift_abs.png")
    print("- plot_shape_drift_rel.png")
    print("- plot_reprojection_error.png")
    print("- plot_inliers.png")
    print("- plot_inlier_ratio.png")
    print("- plot_kps_cur.png")
    print("- plot_good_matches.png")


if __name__ == "__main__":
    main()