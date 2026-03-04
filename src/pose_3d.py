import os
import csv
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


def main(
    video_path="video/person.mp4",
    out_dir="outputs",
    preview=True,
    max_frames=0,  # 0 = без лимита
):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pose_3d_landmarks.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model_path = "models/pose_landmarker.task"

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    pose = vision.PoseLandmarker.create_from_options(options)

    # Matplotlib 3D окно
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("MediaPipe Pose 3D skeleton (relative coords)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Ограничения осей
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    # CSV: frame_idx, landmark_id, x,y,z, visibility
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "x", "y", "z", "visibility"])

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            res = pose.detect_for_video(mp_image, timestamp_ms)

            vis_frame = frame.copy()

            # обновляем 3D-график
            ax.cla()
            ax.set_title("MediaPipe Pose 3D (Tasks)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-0.5, 0.5)

            if res.pose_landmarks:
                lm = res.pose_landmarks[0]  # список landmarks

                xs = np.array([p.x for p in lm], dtype=np.float32)
                ys = np.array([p.y for p in lm], dtype=np.float32)
                zs = np.array([p.z for p in lm], dtype=np.float32)

                # центрируем 
                xs_vis = xs - xs.mean()
                ys_vis = -(ys - ys.mean())
                zs_vis = zs - zs.mean()

                ax.scatter(xs_vis, ys_vis, zs_vis)

                # 2D точки на видео 
                h, w = vis_frame.shape[:2]
                for p in lm:
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 0), -1)

                # CSV
                for i, p in enumerate(lm):
                    vis = getattr(p, "visibility", None)
                    writer.writerow([frame_idx, i, p.x, p.y, p.z, vis])

            if preview:
                cv2.imshow("Pose 2D (points)", vis_frame)
                plt.pause(0.001)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

    print("Готово.")
    print(f"- CSV с 3D точками: {csv_path}")
    print("Координаты относительные (нормализованные), это 3D модель позы (скелет).")


if __name__ == "__main__":
    main()