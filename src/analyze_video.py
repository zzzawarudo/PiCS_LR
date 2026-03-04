import argparse
import os
import csv
import cv2
import numpy as np

from detectors import available_methods, detect_in_roi, match_descriptors
from utils import compute_frame_indices, get_frame_at_index, select_point_on_frame, make_roi


def estimate_homography_and_drift(matches, kps1, kps2, center_xy):
    """
    Пытаемся оценить гомографию по матчам (RANSAC).
    Считаем drift: куда переедет выбранная точка center_xy при переходе ref->tgt.
    """
    if len(matches) < 4:
        return {
            "ok": False,
            "reason": "not_enough_matches",
            "inliers": 0,
            "inlier_ratio": 0.0,
            "drift_px": None,
        }

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None or mask is None:
        return {
            "ok": False,
            "reason": "homography_failed",
            "inliers": 0,
            "inlier_ratio": 0.0,
            "drift_px": None,
        }

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = float(inliers) / float(len(matches)) if matches else 0.0

    # transform center point
    cx, cy = center_xy
    p = np.array([[[cx, cy]]], dtype=np.float32)  # shape (1,1,2)
    p2 = cv2.perspectiveTransform(p, H)[0, 0]  # (x,y)

    drift = float(np.linalg.norm(p2 - np.array([cx, cy], dtype=np.float32)))

    return {
        "ok": True,
        "reason": "ok",
        "H": H,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "drift_px": drift,
        "center_tgt_xy": (float(p2[0]), float(p2[1])),
        "mask": mask,
    }


def draw_roi(frame, roi, center_xy, color=(0, 255, 0)):
    x1, y1, x2, y2 = roi
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.circle(out, center_xy, 6, color, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video/input.mp4", help="Path to input video")
    parser.add_argument("--roi_radius", type=int, default=120, help="ROI radius in pixels")
    parser.add_argument("--ref_sec", type=float, default=0.5, help="Seconds from start for reference frame")
    parser.add_argument("--tgt_sec_from_end", type=float, default=0.5, help="Seconds from end for target frame")
    parser.add_argument("--ratio", type=float, default=0.75, help="Ratio test threshold")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    fps, total_frames, ref_idx, tgt_idx = compute_frame_indices(cap, args.ref_sec, args.tgt_sec_from_end)

    ok, ref_frame = get_frame_at_index(cap, ref_idx)
    if not ok:
        raise RuntimeError("Не удалось считать reference frame")

    ok, tgt_frame = get_frame_at_index(cap, tgt_idx)
    if not ok:
        raise RuntimeError("Не удалось считать target frame")

    print(f"Video: {args.video}")
    print(f"FPS ~ {fps:.2f}, total_frames = {total_frames}")
    print(f"Reference frame index = {ref_idx}, target frame index = {tgt_idx}")

    # выбираем точку на reference кадре
    pt = select_point_on_frame(ref_frame, win_name="Выберите точку на первом кадре")
    if pt is None:
        print("Отмена выбора точки. Выход.")
        return

    h, w = ref_frame.shape[:2]
    roi = make_roi(pt, args.roi_radius, w, h)

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_frame, cv2.COLOR_BGR2GRAY)

    methods = available_methods()
    if not methods:
        raise RuntimeError("Не удалось создать ни один детектор.")

    # сохраним ref/tgt с ROI
    cv2.imwrite(os.path.join(args.out_dir, "ref_with_roi.png"), draw_roi(ref_frame, roi, pt))
    cv2.imwrite(os.path.join(args.out_dir, "tgt.png"), tgt_frame)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "kps_ref",
            "kps_tgt",
            "good_matches",
            "inliers",
            "inlier_ratio",
            "drift_px",
            "status",
        ])

        for m in methods:
            print(f"\n=== {m.name} ===")

            kps1, desc1 = detect_in_roi(m, ref_gray, roi)
            kps2, desc2 = detect_in_roi(m, tgt_gray, roi)

            matches = match_descriptors(m, desc1, desc2, ratio=args.ratio)

            result = estimate_homography_and_drift(matches, kps1, kps2, pt)

            drift_px = result["drift_px"] if result["drift_px"] is not None else ""
            inlier_ratio = result["inlier_ratio"]

            writer.writerow([
                m.name,
                len(kps1),
                len(kps2),
                len(matches),
                result["inliers"],
                f"{inlier_ratio:.3f}",
                drift_px,
                result["reason"],
            ])

            print(f"kps_ref={len(kps1)} kps_tgt={len(kps2)} good_matches={len(matches)}")
            print(f"inliers={result['inliers']} inlier_ratio={inlier_ratio:.3f} drift_px={result['drift_px']} status={result['reason']}")

            # визуализация матчей
            # рисуем только good matches
            match_img = cv2.drawMatches(
                ref_frame, kps1,
                tgt_frame, kps2,
                matches[:200], None,  # ограничим, чтобы картинка не была гигантской
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(os.path.join(args.out_dir, f"matches_{m.name}.png"), match_img)

            # если удалось посчитать центр на target — отметим его
            if result.get("ok"):
                cx2, cy2 = result["center_tgt_xy"]
                vis_tgt = tgt_frame.copy()
                cv2.rectangle(vis_tgt, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.circle(vis_tgt, (int(round(cx2)), int(round(cy2))), 7, (0, 0, 255), 2)
                cv2.putText(vis_tgt, f"{m.name} drift={result['drift_px']:.1f}px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(args.out_dir, f"tgt_center_{m.name}.png"), vis_tgt)

    cap.release()
    print("\nГотово. Результаты в папке outputs")


if __name__ == "__main__":
    main()