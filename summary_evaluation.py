import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from tqdm import tqdm

RESULT_ROOT = './results_llava/results/dataset/all_data/'
ANNOTATIONS_ROOT = './dataset/Annotation/'
IMAGE_ROOT = './dataset/all_data/'


def plot(results, y_axis='accuracy', area_bins=20, contrast_bins=5):
    df = pd.DataFrame(results, columns=['result', 'area', 'weber_ratio'])

    # area 分箱
    df['area_bin'] = pd.qcut(df['area'], q=area_bins)
    df['area_bin_num'] = df['area_bin'].cat.codes
    area_bin_labels = df['area_bin'].cat.categories.astype(str).to_list()

    # weber_ratio 分组
    df['contrast_group'] = pd.qcut(
        df['weber_ratio'],
        q=contrast_bins,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'][:contrast_bins]
    )

    plt.figure(figsize=(10, 6))

    # 画图
    sns.lineplot(
        data=df,
        x='area_bin_num',
        y='result',
        hue='contrast_group',
        estimator='mean',
        errorbar=None
    )

    # 美化横轴
    plt.xticks(
        ticks=range(len(area_bin_labels)),
        labels=area_bin_labels,
        rotation=45,
        ha='right'
    )

    plt.title(f'Accuracy vs Area Bin (Grouped by Contrast - {contrast_bins} Levels)')
    plt.xlabel('Area Bin (box size ranges)')
    plt.ylabel(y_axis)
    plt.legend(title='Contrast Group')
    plt.tight_layout()
    plt.show()


def read_area(annotation_path: Path):
    path = str(annotation_path)
    with open(path, 'r') as f:
        json_gt = json.load(f)
        boxes_gt = json_gt['det_boxes'][0]
        x1, y1, x2, y2 = boxes_gt
        return (x2 - x1) * (y2 - y1)

def compute_luminance_contrast(annotation_path: Path, image_path: Path):

    with open(str(annotation_path), 'r') as f:
        json_gt = json.load(f)
        boxes_gt = json_gt['det_boxes']

    image = cv2.imread(str(image_path))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_diffs = []
    weber_ratios = []

    for x1, y1, x2, y2 in boxes_gt:
        smoke = image_gray[y1:y2, x1:x2]
        smoke_mean = np.mean(smoke)

        pad = 20
        h, w = image_gray.shape
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(w, x2 + pad)
        by2 = min(h, y2 + pad)

        background = image_gray[by1:by2, bx1:bx2].copy()
        background[(y1 - by1):(y2 - by1), (x1 - bx1):(x2 - bx1)] = 0
        background_pixels = background[background > 0]
        if len(background_pixels) == 0:
            continue

        bg_mean = np.mean(background_pixels)

        mean_diff = abs(smoke_mean - bg_mean)
        mean_diffs.append(mean_diff)

        if bg_mean != 0:
            weber_ratio = abs(smoke_mean - bg_mean) / bg_mean
            weber_ratios.append(weber_ratio)

    avg_mean_diff = np.mean(mean_diffs) if mean_diffs else 0
    avg_weber_ratio = np.mean(weber_ratios) if weber_ratios else 0

    return avg_mean_diff, avg_weber_ratio

def classification_negative(results_dir: Path):
    negative_path = results_dir / 'classification-negative'
    json_files = list(negative_path.rglob("*.json"))
    correct = 0
    total = 0
    for json_file in json_files:
        with open(json_file, 'r') as a_f:
            a_f_j = json.load(a_f)
            if a_f_j['prediction'] == a_f_j['label']:
                correct += 1
            total += 1
    print(correct, total)

def classification(results_dir: Path, data_dir: Path, annotation_dir: Path):
    results_path = results_dir / 'classification'
    json_files = list(results_path.rglob("*.json"))
    results = []
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as a_f:
            a_f_j = json.load(a_f)
            result = a_f_j['prediction'] == a_f_j['label']
        image_path = data_dir / json_file.parent.parent.name / json_file.parent.name
        annotation_path = annotation_dir / json_file.parent.parent.name / json_file.parent.name.replace('.jpg', '.json')
        area = read_area(annotation_path)
        mean_contrast, weber_ratio = compute_luminance_contrast(annotation_path, image_path)
        results.append((result, area, weber_ratio))
    plot(results)

def coordinate(results_dir: Path, data_dir: Path, annotation_dir: Path):
    results_path = results_dir / 'coordinate'
    json_files = list(results_path.rglob("*.json"))
    results = []
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as a_f:
            a_f_j = json.load(a_f)
            result = np.max((a_f_j['iou']))
        image_path = data_dir / json_file.parent.parent.name / json_file.parent.name
        annotation_path = annotation_dir / json_file.parent.parent.name / json_file.parent.name.replace('.jpg', '.json')
        area = read_area(annotation_path)
        mean_contrast, weber_ratio = compute_luminance_contrast(annotation_path, image_path)
        results.append((result, area, weber_ratio))
    plot(results)

def grid(results_dir: Path, data_dir: Path, annotation_dir: Path):
    results_path = results_dir / 'grid'
    json_files = list(results_path.rglob("*.json"))
    results = []
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as a_f:
            a_f_j = json.load(a_f)
            result = a_f_j['iou']
        image_path = data_dir / json_file.parent.parent.name / json_file.parent.name
        annotation_path = annotation_dir / json_file.parent.parent.name / json_file.parent.name.replace('.jpg', '.json')
        area = read_area(annotation_path)
        mean_contrast, weber_ratio = compute_luminance_contrast(annotation_path, image_path)
        results.append((result, area, weber_ratio))
    plot(results)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--annotation_dir', type=str)
    parser.add_argument('--task_type', type=str,
                        choices=['classification', 'classification-negative', 'grid', 'coordinate'],
                        default='grid')
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_dir)
    data_path = Path(args.data_dir)
    annotation_path = Path(args.annotation_dir)
    if args.task_type == 'classification':
        classification(results_path, data_path, annotation_path)
    elif args.task_type == 'classification-negative':
        classification_negative(results_path)
    elif args.task_type == 'grid':
        grid(results_path, data_path, annotation_path)
    elif args.task_type == 'coordinate':
        coordinate(results_path, data_path, annotation_path)
    else:
        raise ValueError


if __name__ == '__main__':
    main()
