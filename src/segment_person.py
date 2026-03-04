import os
import cv2
import numpy as np
import mediapipe as mp

def main(
    video_path="video/person.mp4",
    out_dir="outputs",
    preview=True,
    mask_threshold=0.5,
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mask = cv2.VideoWriter(os.path.join(out_dir, "person_mask.mp4"), fourcc, fps, (w, h))
    out_cut = cv2.VideoWriter(os.path.join(out_dir, "person_cut.mp4"), fourcc, fps, (w, h))
    out_overlay = cv2.VideoWriter(os.path.join(out_dir, "person_overlay.mp4"), fourcc, fps, (w, h))

    # Новый API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = "models/selfie_multiclass_256x256.tflite"

    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(frame_idx * (1000.0 / fps))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = segmenter.segment_for_video(mp_image, timestamp_ms)

            category_mask = result.category_mask.numpy_view()
            if category_mask.ndim == 3:
                category_mask = category_mask[:, :, 0]

            mask = (category_mask != 0).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            person = (mask != 0)  # (H,W) bool

            cut = frame.copy()
            cut[~person] = (0, 0, 0)

            overlay = frame.copy()
            green = np.zeros_like(frame)
            green[:] = (0, 255, 0)
            alpha = 0.35
            overlay[person] = (overlay[person] * (1 - alpha) + green[person] * alpha).astype(np.uint8)

            out_mask.write(mask_bgr)
            out_cut.write(cut)
            out_overlay.write(overlay)

            if preview:
                cv2.imshow("Mask", mask_bgr)
                cv2.imshow("Cut", cut)
                cv2.imshow("Overlay", overlay)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            frame_idx += 1

    cap.release()
    out_mask.release()
    out_cut.release()
    out_overlay.release()
    cv2.destroyAllWindows()

    print("Готово. Файлы в outputs")


if __name__ == "__main__":
    main()