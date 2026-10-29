# Orange Disease Classification — Greenpod Labs Assignment 2

A computer vision pipeline to classify orange conditions into four categories:
**Blackspot**, **Canker**, **Fresh**, and **Grenning**.

Two approaches are implemented and compared:
- **Approach 1:** Classical CV — HSV/LAB color features + LBP texture → SVM + Random Forest
- **Approach 2:** Deep Learning — MobileNetV2 transfer learning (PyTorch)

## DatasetLink: https://drive.google.com/drive/folders/1XmAfp0hdoN6sC7BiccO0Kr07jsJT0rfG?usp=drive_link
---

## Dataset Structure

```
data/raw/
├── train/
│   ├── blackspot/   (184 images)
│   ├── canker/      (179 images)
│   ├── fresh/       (281 images)
│   └── grenning/    (347 images)
└── test/
    ├── blackspot/   (22 images)
    ├── canker/      (22 images)
    ├── fresh/       (33 images)
    └── grenning/    (22 images)
```

---

## Setup

```bash
git clone <repo-url>
cd orange_classification
pip install -r requirements.txt
```

Place the unzipped dataset into `data/raw/` so you have `data/raw/train/` and `data/raw/test/`.

---

## Execution Order

Run the notebooks in order:

```
notebooks/01_eda_and_data_cleaning.ipynb
notebooks/02_approach_1_classical_ml.ipynb
notebooks/03_approach_2_deep_learning.ipynb
notebooks/04_model_evaluation_and_demo.ipynb
```

Or run the demo script directly (after training both models):

```bash
python main_demo.py --image data/raw/test/fresh/f\ \(1\).png
```

---

## Results Summary

| Model              | Accuracy | Macro F1 |
|--------------------|----------|----------|
| SVM (Classical)    | ~82%     | ~0.81    |
| Random Forest      | ~79%     | ~0.78    |
| MobileNetV2 (DL)   | ~95%     | ~0.94    |

*Exact numbers depend on your hardware and random seed. Run notebook 04 to reproduce.*

---

## Project Structure

```
orange_classification/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Unzipped dataset
│   └── processed/            # Saved train/val/test splits (numpy arrays)
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_approach_1_classical_ml.ipynb
│   ├── 03_approach_2_deep_learning.ipynb
│   └── 04_model_evaluation_and_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── classical_model.py
│   └── dl_model.py
├── models/                   # Saved model weights
├── main_demo.py
└── report/
    └── Assignment2_Orange_Report.pdf
```
