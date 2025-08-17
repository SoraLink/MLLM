import argparse
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
from mllms import MLLM_LLAVA, InternVL3, UIO2, QwenVL, GroundingDINO

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Evaluator:

    def __init__(self):
        self.prompt_bbox = (
            'Detect all smoke and output bounding box like'
            '[[x1, y1, x2, y2], [x1, y1, x2, y2]] '
            "If you cannot find any smoke return empty list [] without other words"
            "Do not return dict!"
        )

        self.prompt_grid = (
            'Please look at this image, which is divided into several numbered regions '
            '(from left to right, top to bottom). '
            'Please output the numbered regions that contain smoke in JSON format as '
            'a list of dicts like [{"region": 1}, {"region": 2}]. '
            "If you cannot find any smoke return empty list [] without other words"
        )

        self.prompt_classification = (
            'Please look at this image. Detect if the image contains smoke. '
            'If you can find any smoke, return True. Otherwise, return False.'
        )

    def evaluate(self, image_path, annotation_path, mode, result_root):
        if self._has_result(result_root, image_path, mode):
            print('Skip {} because it is already evaluated'.format(annotation_path))
            return
        if mode == 'coordinate':
            result = self._evaluate_coordinate(image_path, annotation_path)
        elif mode == 'grid':
            result = self._evaluate_grid(image_path, annotation_path)
        elif mode == 'coordinate_by_grid':
            result = self._evaluate_coordinate_by_grid(image_path, annotation_path)
        elif mode == 'classification':
            result = self._evaluate_classification(image_path)
        elif mode == 'subimage_classification':
            result = self._evaluate_subimage_classification(image_path, annotation_path)
        else:
            raise ValueError('No such mode: {}'.format(mode))
        self._save_result(result_root, result, image_path, mode)

    def _evaluate_subimage_classification(self, image_path, annotation_path):
        image = cv2.imread(image_path)
        h, w, c = image.shape

        h_block = h // 3
        w_block = w // 4

        sub_images = []

        for i in range(3):
            for j in range(4):
                y1, y2 = i * h_block, (i + 1) * w_block
                x1, x2 = j * w_block, (j + 1) * h_block
                sub_image = image[y1:y2, x1:x2]
                sub_images.append(sub_image)
        results = self.model.batch_predict(sub_images, self.prompt_classification)
        predict_grids = []
        for i, result in enumerate(results):
            if 'True' in result or 'true' in result:
                predict_grids.append(i+1)
        annotation_grids = get_annotation_grid_number(annotation_path, image, rows=3, cols=4)
        iou = compute_grid_IoU(set(predict_grids), set(annotation_grids))
        return {
            'iou': iou,
            'predict_grids': predict_grids,
            'annotation_grids': annotation_grids,
            'response': results
        }



    def _evaluate_classification(self, image_path):
        image = cv2.imread(image_path)
        response = self.model.predict(image, self.prompt_classification)
        prediction = True if 'True' in response else False
        image_name = Path(image_path).name
        label = image_name.split('_')[1][0] == '+'
        return {
            'prediction': prediction,
            'response': response,
            'image_name': image_name,
            'label': label
        }

    def _evaluate_coordinate(self, image_path, annotation_path):
        image = cv2.imread(image_path)
        prediction = self.model.predict(image, self.prompt_bbox)
        bboxes = self._retireve_bbox(prediction)
        with open(annotation_path, 'r') as f:
            json_gt = json.load(f)
            boxes_gt = json_gt['det_boxes']
            pred_boxes = torch.tensor(bboxes)
            gt_boxes = torch.tensor(boxes_gt)
            ious = box_iou(pred_boxes, gt_boxes)
            return {
                'iou': ious.tolist(),
                'predict_boxes': pred_boxes.tolist(),
                'gt_boxes': gt_boxes.tolist(),
                'response': prediction,
                'img': image
            }

    def _evaluate_coordinate_by_grid(self, image_path, annotation_path):
        img_rgb = cv2.imread(image_path)
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
            boxes = json_data['det_boxes']
        if len(boxes) > 1:
            raise NotImplementedError('Multiple')
        pred_box, iou, responses, imgs = self._grid_search(img_rgb, boxes[0])
        return {
            'iou': iou,
            'predict_box': pred_box,
            'gt_box': boxes[0],
            'responses': responses,
            'img': imgs
        }


    def _grid_search(self, image, gt_box, patch_box=None):
        if patch_box is None:
            height, width = image.shape[:2]
            patch_box = (0, 0, width, height)
        cut_img = ImagePreprocess.add_grid_to_patch(image, patch_box)
        content = self.model.predict(cut_img, self.prompt_grid)
        predict_grids = self._retrieve_grid_number(content)
        pred_box = self._region_ids_to_box(predict_grids, patch_box)
        ious = box_iou(torch.tensor([pred_box]), torch.tensor([gt_box]))
        pred_box_area = self._box_area(pred_box)
        bbox_area = self._box_area(gt_box)
        if ious.item() > 0.5 or pred_box_area < bbox_area or pred_box == gt_box:
            return pred_box, ious.item(), [content], [cut_img]
        pred_box, iou, responses, imgs = self._grid_search(image, gt_box, pred_box)
        responses.insert(0, content)
        imgs.insert(0, cut_img)
        return pred_box, iou, responses, imgs



    def _region_ids_to_box(self, region_ids, patch_box, rows=5, cols=5):
        if len(region_ids) == 0:
            return (0, 0, 0, 0)
        x1, y1, x2, y2 = patch_box
        cell_w = (x2 - x1) // cols
        cell_h = (y2 - y1) // rows

        boxes = []
        for rid in region_ids:
            row = (rid - 1) // cols
            col = (rid - 1) % cols

            bx1 = x1 + col * cell_w
            by1 = y1 + row * cell_h
            bx2 = bx1 + cell_w
            by2 = by1 + cell_h
            boxes.append((bx1, by1, bx2, by2))

        bx1s = [b[0] for b in boxes]
        by1s = [b[1] for b in boxes]
        bx2s = [b[2] for b in boxes]
        by2s = [b[3] for b in boxes]

        merged_bbox = (
            min(bx1s),
            min(by1s),
            max(bx2s),
            max(by2s)
        )

        return merged_bbox


    def _box_area(self, box):
        width = abs(box[2] - box[0])
        height = abs(box[3] - box[1])
        return width * height


    def _evaluate_grid(self, image_path, annotation_path):
        img_rgb = ImagePreprocess.add_grid(image_path)
        annotation_grids = get_annotation_grid_number(annotation_path, img_rgb)

        content = self.model.predict(img_rgb, self.prompt_grid)
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
            bboxes = json.loads(response_content)
            if len(bboxes) == 0:
                return [[0, 0, 0, 0]]
            bboxes = [[0, 0, 0, 0] if len(box) == 0 else box for box in bboxes]
        except Exception:
            print(response_content)
            bboxes = [[0, 0, 0, 0]]
        return bboxes

    def _save_result(self, path: str, evaluation_result: dict, image_path: str, mode: str):
        image_path = Path(image_path)
        path = Path(os.path.join(path, image_path.parent.name, image_path.name)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if 'img' in evaluation_result:
            img = evaluation_result.pop('img')
            if isinstance(img, list):
                for idx, single_img in enumerate(img):
                    filename = 'img-{}-{}.jpg'.format(mode, idx)
                    cv2.imwrite(str(path / filename), single_img)
            else:
                cv2.imwrite(str(path / 'img_{}.jpg'.format(mode)), img)
        json_path = path / 'results_{}.json'.format(mode)
        print(evaluation_result)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f)

    def _has_result(self, path: str, image_path: str, mode: str):
        image_path = Path(image_path)
        path = Path(os.path.join(path, image_path.parent.name, image_path.name)).resolve()
        json_path = path / 'results_{}.json'.format(mode)
        if json_path.exists():
            return True
        return False


class LlavaEvaluation(Evaluator):

    def __init__(self):
        super().__init__()
        self.model = MLLM_LLAVA.get_instance()


class InternVL3Evaluation(Evaluator):

    def __init__(self):
        super().__init__()
        self.model = InternVL3.get_instance()


class UIO2Evaluation(Evaluator):

    def __init__(self):
        super().__init__()
        self.prompt_bbox = (
            "smoke"
        )
        self.model = UIO2.get_instance()

class QWen2VLEvaluation(Evaluator):
    def __init__(self):
        super().__init__()
        self.model = QwenVL.get_instance()

class GroundingDinoEvaluation(Evaluator):

    def __init__(self):
        super().__init__()
        self.model = GroundingDINO()

def build_evaluation(model):
    if model == 'InternVL3':
        return InternVL3Evaluation()
    elif model == 'Llava':
        return LlavaEvaluation()
    elif model == 'uio2':
        return UIO2Evaluation()
    elif model == 'QwenVL':
        return QWen2VLEvaluation()
    elif model == 'GroundingDino':
        return GroundingDinoEvaluation()
    else:
        raise NotImplementedError

def evaluate(annotation_path, dataset_path, result_path, model, evaluation_type, is_negative):
    evaluation = build_evaluation(model)
    if is_negative:
        with open(dataset_path, 'r') as f:
            for line in f.readlines():
                path = line.strip()
                print('evaluating image {}'.format(path))
                evaluation.evaluate(path, None, mode=evaluation_type, result_root=result_path)

    else:
        root_dir = Path(annotation_path)
        for folder in tqdm(root_dir.iterdir()):
            if folder.is_dir():
                for label in folder.iterdir():
                    if label.name.endswith('.json'):
                        image_name = label.name.replace('.json', '.jpg')
                        image_path = os.path.join(dataset_path, folder.name, image_name)
                        annotation_path = str(label)
                        print('evaluating image {}'.format(image_path))
                        evaluation.evaluate(image_path, annotation_path, mode=evaluation_type, result_root=result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset/all_data')
    parser.add_argument('--is_negative', type=bool, default=False)
    parser.add_argument('--annotations', type=str, default="./dataset/Annotation")
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--evaluation_type', type=str, default='grid', choices=['grid', 'coordinate', 'coordinate_by_grid', 'classification', 'subimage_classification'])
    parser.add_argument('--model', type=str, default='InternVL3', choices=['InternVL3', 'Llava', 'uio2', 'QwenVL', 'GroundingDino'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.annotations, args.dataset, args.output_dir, args.model, args.evaluation_type, args.is_negative)
