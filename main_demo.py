import argparse
import os
import sys
import numpy as np
import torch
import joblib
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from src.classical_model import extract_features_batch
from src.dl_model import build_mobilenet, predict_single

CLASS_NAMES = ['blackspot', 'canker', 'fresh', 'grenning']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_models(models_dir='models'):
    svm_path = os.path.join(models_dir, 'svm_model.joblib')
    rf_path  = os.path.join(models_dir, 'rf_model.joblib')
    dl_path  = os.path.join(models_dir, 'mobilenet_best.pth')

    for p in [svm_path, rf_path, dl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model not found: {p}. Run the notebooks first.")

    svm = joblib.load(svm_path)
    rf  = joblib.load(rf_path)

    dl_model = build_mobilenet(num_classes=4).to(DEVICE)
    dl_model.load_state_dict(torch.load(dl_path, map_location=DEVICE))
    dl_model.eval()

    return svm, rf, dl_model


def run_demo(image_path, models_dir='models'):
    svm, rf, dl_model = load_models(models_dir)

    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img.resize((224, 224)))

    feat = extract_features_batch(np.expand_dims(img_np, 0))
    svm_pred = CLASS_NAMES[svm.predict(feat)[0]]
    rf_pred  = CLASS_NAMES[rf.predict(feat)[0]]

    tensor = TRANSFORM(img)
    dl_pred, dl_conf = predict_single(dl_model, tensor, DEVICE, CLASS_NAMES)

    print(f"\n--- Prediction Results for: {image_path} ---")
    print(f"  Approach 1 (SVM):        {svm_pred}")
    print(f"  Approach 1 (Random Forest): {rf_pred}")
    print(f"  Approach 2 (MobileNetV2):   {dl_pred}  (confidence: {dl_conf:.1%})")
    print("---------------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orange Disease Classifier Demo')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--models', default='models', help='Directory containing saved models')
    args = parser.parse_args()
    run_demo(args.image, args.models)
