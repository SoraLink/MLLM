import json
import os
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm

RESULT_ROOT = './results/'

def compute_IoU(prediction: set, annotation: set):
    prediction = set([int(n) for n in prediction])
    intersection = prediction.intersection(annotation)
    union = prediction.union(annotation)
    IoU = len(intersection) / len(union)
    return IoU

def summary_evaluation(folder: Path):
    times = []
    ious = []
    for f in folder.iterdir():
        if not f.is_dir():
            continue
        f_name = f.name
        time = f_name.split('+')[1]
        time = time.replace('.jpg', '')
        times.append(time)
        path = f.resolve()
        result_path = os.path.join(path, 'results.json')
        with open(result_path, 'r') as f:
            j = json.load(f)
            prediction_grids = j['predict_grids']
            annotation_grids = j['annotation_grids']
            iou = compute_IoU(prediction_grids, annotation_grids)
            ious.append(iou)
    plt.plot(times, ious)
    plt.xlabel('time')
    plt.ylabel('iou')
    plt.xticks(times[::5], rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(folder.resolve(), 'summary.png'))
    plt.close()


def main():
    root_dir = Path(RESULT_ROOT)
    for folder in tqdm(root_dir.iterdir()):
        if folder.is_dir():
            summary_evaluation(folder)


if __name__ == '__main__':
    main()
