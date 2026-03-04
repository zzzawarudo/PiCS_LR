import cv2
import numpy as np


class Method:
    def __init__(self, name: str, detector, norm_type: int, is_binary: bool):
        self.name = name
        self.detector = detector
        self.norm_type = norm_type
        self.is_binary = is_binary


def available_methods():
    methods = []

    # SIFT
    try:
        sift = cv2.SIFT_create()
        methods.append(Method("SIFT", sift, cv2.NORM_L2, is_binary=False))
    except Exception:
        pass

    # ORB
    try:
        orb = cv2.ORB_create(nfeatures=2000)
        methods.append(Method("ORB", orb, cv2.NORM_HAMMING, is_binary=True))
    except Exception:
        pass

    # AKAZE
    try:
        akaze = cv2.AKAZE_create()
        methods.append(Method("AKAZE", akaze, cv2.NORM_HAMMING, is_binary=True))
    except Exception:
        pass

    return methods


def detect_in_roi(method: Method, gray: np.ndarray, roi):
    """
    Ищем keypoints+descriptors только в ROI.
    Возвращаем keypoints в координатах полного кадра (offset добавляем).
    """
    x1, y1, x2, y2 = roi
    roi_img = gray[y1:y2, x1:x2]
    if roi_img.size == 0:
        return [], None

    kps, desc = method.detector.detectAndCompute(roi_img, None)

    # переводим keypoints в координаты исходного кадра
    kps_full = []
    for kp in kps:
        kp2 = cv2.KeyPoint(
            x=kp.pt[0] + x1,
            y=kp.pt[1] + y1,
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id,
        )
        kps_full.append(kp2)

    return kps_full, desc


def match_descriptors(method: Method, desc1, desc2, ratio: float = 0.75):
    """
    KNN + ratio test. Для бинарных и float дескрипторов одинаково работает.
    """
    if desc1 is None or desc2 is None:
        return []

    if len(desc1) < 2 or len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(method.norm_type, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    return good