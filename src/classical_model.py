import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def extract_features(image_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(image_rgb, (128, 128))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    color_feats = []
    for ch_idx, space in [(0, lab), (1, lab), (2, lab), (0, hsv), (1, hsv), (2, hsv)]:
        hist = cv2.calcHist([space], [ch_idx], None, [16], [0, 256]).flatten()
        color_feats.append(hist / (hist.sum() + 1e-7))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 12), density=True)

    return np.concatenate(color_feats + [lbp_hist])


def extract_features_batch(images: np.ndarray) -> np.ndarray:
    return np.array([extract_features(img) for img in images])


def build_svm():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])


def build_random_forest():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
