import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


class DayNightClassifier:

    def __init__(self):
        self.day_night_classifier = pipeline(
            "image-classification",
            model="Woleek/day-night",
            device=0
        )

    def classify(self, image_path):
        return self.day_night_classifier(image_path)[0]['label']

class GrayRGBClassifier:

    def classify(self, image_path, tol=5, ratio=0.99):
        arr = np.asarray(Image.open(image_path).convert("RGB"))
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2])
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])

        # 计算在 tol 以内的像素比例
        ok = (diff_rg < tol) & (diff_rb < tol) & (diff_gb < tol)
        return 'gray' if ok.mean() >= ratio else 'rgb'
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Output file name")
    parser.add_argument("--classifier", type=str, required=True, choices=["day_night", "grayscale"])
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = args.output_file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.classifier == "day_night":
        classifier = DayNightClassifier()
    elif args.classifier == "grayscale":
        classifier = GrayRGBClassifier()
    else:
        raise ValueError(f"Invalid classifier: {args.classifier}")
    iterate_all(args.folder_path, output_path, classifier)

def iterate_all(folder_path, output_path, classifer):
    with os.scandir(folder_path) as it:
        for entry in tqdm(it):
            if entry.is_dir():
                images = [
                    f for f in os.scandir(entry.path)
                    if f.is_file() and f.name.endswith((".jpg", ".jpeg", ".png"))
                ]
                if not images:
                    continue

                image = random.sample(images, 1)[0]
                label = classifer.classify(image.path)
                save_path = os.path.join(output_path, label)
                os.makedirs(save_path, exist_ok=True)
                shutil.copytree(entry, str(os.path.join(str(save_path), entry.name)), dirs_exist_ok=True)


if __name__ == "__main__":
    main()