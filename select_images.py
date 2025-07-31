import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

ANNOTATION_ROOT = "./dataset/Annotation"
DATA_ROOT = "./dataset/all_data"
AREA_DIR = "./dataset/area.json"
CONTRAST_DIR = "./dataset/contrast.json"

PATH_TO_AREA = {}
PATH_TO_CONTRAST = {}


def read_area(annotation_path):
    path = str(annotation_path)
    with open(path, 'r') as f:
        json_gt = json.load(f)
        boxes_gt = json_gt['det_boxes'][0]
        x1, y1, x2, y2 = boxes_gt
        return (x2 - x1) * (y2 - y1)


def compute_contrast(annotation_path: Path, folder_name: str):
    image_name = annotation_path.name.replace('.json', '.jpg')
    image_path = os.path.join(DATA_ROOT, folder_name, image_name)
    with open(str(annotation_path), 'r') as f:
        json_gt = json.load(f)
        boxes_gt = json_gt['det_boxes']
    image = cv2.imread(image_path)
    contrast_diffs = []
    for x1, y1, x2, y2 in boxes_gt:
        smoke = image[y1:y2, x1:x2]
        smoke_std = np.std(cv2.cvtColor(smoke, cv2.COLOR_BGR2GRAY))

        pad = 20
        h, w = image.shape[:2]
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(w, x2 + pad)
        by2 = min(h, y2 + pad)

        background = image[by1:by2, bx1:bx2].copy()
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        background_gray[(y1 - by1):(y2 - by1), (x1 - bx1):(x2 - bx1)] = 0

        bg_std = np.std(background_gray[background_gray > 0])
        contrast_diffs.append(abs(smoke_std - bg_std))
    avg_contrast_diff = np.mean(contrast_diffs)
    return avg_contrast_diff

def split_dict_into_groups(d, group_size=10):
    sorted_items = sorted(d.items(), key=lambda x: x[1])  # 按数值排序
    total = len(sorted_items)

    first_group = sorted_items[:group_size]

    mid_start = max(0, (total // 2) - (group_size // 2))
    middle_group = sorted_items[mid_start:mid_start + group_size]

    last_group = sorted_items[-group_size:]

    return {
        "lower": first_group,
        "mide": middle_group,
        "upper": last_group
    }

def save_groups(path, groups):
    with open(path, 'w') as f:
        f.write(json.dumps(groups))

def main():
    annotation_root = Path(ANNOTATION_ROOT)
    for folder in annotation_root.iterdir():
        if folder.is_dir():
            for annotation in folder.iterdir():
                area = read_area(annotation)
                PATH_TO_AREA[str(annotation)] = area
                contrast = compute_contrast(annotation, folder.name)
                PATH_TO_CONTRAST[str(annotation)] = contrast
    area_groups = split_dict_into_groups(PATH_TO_AREA)
    contrast_groups = split_dict_into_groups(PATH_TO_CONTRAST)
    save_groups(AREA_DIR, area_groups)
    save_groups(CONTRAST_DIR, contrast_groups)


if __name__ == '__main__':
    main()
