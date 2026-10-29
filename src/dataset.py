import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ['blackspot', 'canker', 'fresh', 'grenning']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_SIZE = 224


def load_images_from_folder(folder, img_size=IMG_SIZE):
    images, labels = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(class_dir, fname)
            try:
                img = Image.open(path).convert('RGB').resize((img_size, img_size))
                images.append(np.array(img))
                labels.append(CLASS_TO_IDX[class_name])
            except Exception:
                pass
    return np.array(images), np.array(labels)


def get_train_val_split(train_dir, val_size=0.15, random_state=42):
    X, y = load_images_from_folder(train_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


class OrangeDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
