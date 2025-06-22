import json
import os
import re
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from mllm_smoke_locate import ImagePreprocess, get_annotation_grid_number, compute_grid_IoU, add_bbox
from mllms import MLLM_LLAVA

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LlavaEvaluation:

    def __init__(self):
        self.llava_model = MLLM_LLAVA.get_instance()

    def evaluate(self, image_path, annotation_path, mode):
        if mode == 'coordinate':
            result = self._evaluate_coordinate(image_path, annotation_path)
        elif mode == 'grid':
            result = self._evaluate_grid(image_path, annotation_path)
        else:
            raise ValueError('No such mode: {}'.format(mode))
        self._save_result(RESULT_ROOT, result, image_path, mode)

    def _evaluate_coordinate(self, image_path, annotation_path):
        image = cv2.imread(image_path)
        prediction = self.llava_model.predict(image)
        bboxes = self._retireve_bbox(prediction)
        with open(annotation_path, 'r') as f:
            json_gt = json.load(f)
            boxes_gt = json_gt['det_boxes']
            pred_boxes = torch.tensor(bboxes)
            gt_boxes = torch.tensor(boxes_gt)
            ious = box_iou(pred_boxes, gt_boxes)
            return {
                'iou': ious.cpu().numpy(),
                'response': prediction,
                'img': image
            }

    def _evaluate_grid(self, image_path, annotation_path):
        img_rgb = ImagePreprocess.add_grid(image_path)
        annotation_grids = get_annotation_grid_number(annotation_path, img_rgb)

        content = self.llava_model.predict(img_rgb, is_grid=True)
        predict_grids = self._retrieve_grid_number(content)
        iou = compute_grid_IoU(set(predict_grids), set(annotation_grids))
        img_with_box = add_bbox(annotation_path, img_rgb)
        return {
            'iou': iou,
            'predict_grids': predict_grids,
            'annotation_grids': annotation_grids,
            'img': img_with_box,
            'response': content
        }

    def _retrieve_grid_number(self, response_content):
        numbers = re.findall(r'\d+', response_content)
        numbers = [int(number) for number in numbers]
        return numbers

    def _retireve_bbox(self, response_content):
        data_list = json.loads(response_content)
        bboxes = [tuple(item['bbox']) for item in data_list]
        return bboxes

    def _save_result(self, path: str, evaluation_result: dict, image_path: str, mode: str):
        path = Path(os.path.join(path, image_path)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        img = evaluation_result.pop('img')
        json_path = path / 'results_{}.json'.format(mode)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f)
        cv2.imwrite(str(path / 'img_{}.jpg'.format(mode)), img)


ANNOTATION_ROOT = "./dataset/Annotation"
DATA_ROOT = "./dataset/all_data"
RESULT_ROOT = "./results"

def main():
    root_dir = Path(ANNOTATION_ROOT)
    all_ious = []
    evaluation = LlavaEvaluation()
    for folder in tqdm(root_dir.iterdir()):
        if folder.is_dir():
            for label in folder.iterdir():
                if label.name.endswith('.json'):
                    image_name = label.name.replace('.json', '.jpg')
                    image_path = os.path.join(DATA_ROOT, folder.name, image_name)
                    annotation_path = str(label)
                    evaluation.evaluate(image_path, annotation_path, mode='grid')
                    evaluation.evaluate(image_path, annotation_path, mode='coordinate')
    mean_iou = sum(all_ious) / len(all_ious)
    print(mean_iou)

if __name__ == "__main__":
    main()
