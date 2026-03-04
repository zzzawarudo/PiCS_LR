import cv2

path = "video/input.mp4"
cap = cv2.VideoCapture(path)

print("Opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

ok, frame = cap.read()
print("Read first frame:", ok, "shape:", None if frame is None else frame.shape)

cap.release()