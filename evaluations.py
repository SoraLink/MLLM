import ast
import json
import os
import re
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from mllm_smoke_locate import ImagePreprocess, get_annotation_grid_number, compute_grid_IoU, add_bbox
from mllms import MLLM_LLAVA, InternVL3, UIO2

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


class InternVL3Evaluation:

    def __init__(self):
        self.model = InternVL3.get_instance()

    def evaluate(self, image_path, annotation_path, mode, result_root):
        if self._has_result(result_root, image_path, mode):
            print('Skip {} because it is already evaluated'.format(annotation_path))
            return
        if mode == 'coordinate':
            result = self._evaluate_coordinate(image_path, annotation_path)
        elif mode == 'grid':
            result = self._evaluate_grid(image_path, annotation_path)
        else:
            raise ValueError('No such mode: {}'.format(mode))
        self._save_result(result_root, result, image_path, mode)

    def _evaluate_coordinate(self, image_path, annotation_path):
        image = cv2.imread(image_path)
        prediction = self.model.predict(image)
        prediction = ast.literal_eval(prediction)
        with open(annotation_path, 'r') as f:
            json_gt = json.load(f)
            boxes_gt = json_gt['det_boxes']
            pred_boxes = torch.tensor(prediction).view(-1, 4)
            gt_boxes = torch.tensor(boxes_gt)
            ious = box_iou(pred_boxes, gt_boxes).tolist()
            return {
                'iou': ious,
                'predict_boxes': pred_boxes.tolist(),
                'gt_boxes': gt_boxes.tolist(),
                'response': prediction,
                'img': image
            }

    def _evaluate_grid(self, image_path, annotation_path):
        img_rgb = ImagePreprocess.add_grid(image_path)
        annotation_grids = get_annotation_grid_number(annotation_path, img_rgb)

        content = self.model.predict(img_rgb, is_grid=True)
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
        try:
            data_list = json.loads(response_content)
            bboxes = [tuple(item['bbox']) for item in data_list]
        except Exception:
            print(response_content)
            bboxes = []
        return bboxes

    def _save_result(self, path: str, evaluation_result: dict, image_path: str, mode: str):
        path = Path(os.path.join(path, image_path)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        img = evaluation_result.pop('img')
        json_path = path / 'results_{}.json'.format(mode)
        print(evaluation_result)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f)
        cv2.imwrite(str(path / 'img_{}.jpg'.format(mode)), img)

    def _has_result(self, path: str, image_path: str, mode: str):
        path = Path(os.path.join(path, image_path)).resolve()
        json_path = path / 'results_{}.json'.format(mode)
        if json_path.exists():
            return True
        return False


class UIO2Evaluation(InternVL3Evaluation):

    def __init__(self):
        super().__init__()
        self.model = UIO2.get_instance()


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


def area_evaluation(evaluation):
    with open(os.path.join('./dataset/area.json')) as f:
        json_data = json.load(f)
        lower_group = json_data['lower']
        mid_group = json_data['mide']
        upper_group = json_data['upper']
        print('testing low group')
        for path, _ in lower_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'low'))
            evaluation.evaluate(image_path, annotation_path, mode='grid', result_root=os.path.join(RESULT_ROOT, 'low'))
        print('testing mid group')
        for path, _ in mid_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='grid', result_root=os.path.join(RESULT_ROOT, 'mid'))
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'mid'))
        print('testing up group')
        for path, _ in upper_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='grid', result_root=os.path.join(RESULT_ROOT, 'up'))
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'up'))


def contrast_evaluation(evaluation):
    with open(os.path.join('./dataset/contrast.json')) as f:
        json_data = json.load(f)
        lower_group = json_data['lower']
        mid_group = json_data['mide']
        upper_group = json_data['upper']
        print('testing low group')
        for path, _ in lower_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='grid',
                                result_root=os.path.join(RESULT_ROOT, 'low'))
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'low'))
        print('testing mid group')
        for path, _ in mid_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='grid',
                                result_root=os.path.join(RESULT_ROOT, 'mid'))
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'mid'))
        print('testing up group')
        for path, _ in upper_group:
            annotation_path = Path(path)
            image_name = annotation_path.name.replace('.json', '.jpg')
            image_path = os.path.join(DATA_ROOT, annotation_path.parent.name, image_name)
            print('testing', image_path)
            evaluation.evaluate(image_path, annotation_path, mode='grid',
                                result_root=os.path.join(RESULT_ROOT, 'up'))
            evaluation.evaluate(image_path, annotation_path, mode='coordinate',
                                result_root=os.path.join(RESULT_ROOT, 'up'))


if __name__ == "__main__":
    evaluation = UIO2Evaluation()
    area_evaluation(evaluation)
