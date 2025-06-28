import ast
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.ops import box_iou
from tqdm import tqdm

RESULT_ROOT = './results_llava/results/dataset/all_data/'
ANNOTATIONS_ROOT = './dataset/Annotation/'
IMAGE_ROOT = './dataset/all_data/'


def compute_grid_IoU(prediction: set, annotation: set):
    prediction = set([int(n) for n in prediction])
    intersection = prediction.intersection(annotation)
    union = prediction.union(annotation)
    IoU = len(intersection) / len(union)
    return IoU


def compute_IoU(prediction: list, annotation: tuple, image_shape):
    prediction = [
        prediction[0] * image_shape[0],
        prediction[1] * image_shape[1],
        prediction[2] * image_shape[0],
        prediction[3] * image_shape[1]
    ]
    iou = box_iou(torch.tensor([prediction], dtype=torch.float32), torch.tensor(annotation, dtype=torch.float32))
    return iou.item()


def summary_evaluation(folder: Path):
    times = []
    grid_ious = []
    ious = []
    for f in folder.iterdir():
        if not f.is_dir():
            continue
        f_name = f.name
        time = f_name.split('+')[1]
        time = time.replace('.jpg', '')
        times.append(time)
        path = f.resolve()
        grid_result_path = os.path.join(path, 'results_grid.json')
        coordinate_result_path = os.path.join(path, 'results_coordinate.json')
        with open(grid_result_path, 'r') as f:
            j = json.load(f)
            prediction_grids = j['predict_grids']
            annotation_grids = j['annotation_grids']
            iou = compute_grid_IoU(prediction_grids, annotation_grids)
            grid_ious.append(iou)

        with open(coordinate_result_path, 'r') as f:
            j = json.load(f)
            try:
                prediction_bbox = ast.literal_eval(j['response'])
            except Exception:
                prediction_bbox = [0, 0, 0, 0]
            annotation_path = os.path.join(ANNOTATIONS_ROOT, folder.name, f_name.replace('.jpg', '.json'))
            image_path = os.path.join(IMAGE_ROOT, folder.name, f_name)
            image = cv2.imread(image_path)
            image_shape = image.shape
            with open(annotation_path, 'r') as a_f:
                a_f_j = json.load(a_f)
                annotation_bbox = a_f_j['det_boxes']
                iou = compute_IoU(prediction_bbox, annotation_bbox, image_shape)
                ious.append(iou)
    plt.plot(times, grid_ious)
    plt.xlabel('time')
    plt.ylabel('iou')
    plt.xticks(times[::5], rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(folder.resolve(), 'grid_summary.png'))
    plt.close()

    plt.plot(times, ious)
    plt.xlabel('time')
    plt.ylabel('iou')
    plt.xticks(times[::5], rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(folder.resolve(), 'coordinate_summary.png'))
    plt.close()


def main():
    root_dir = Path(RESULT_ROOT)
    for folder in tqdm(root_dir.iterdir()):
        if folder.is_dir():
            summary_evaluation(folder)


if __name__ == '__main__':
    main()
