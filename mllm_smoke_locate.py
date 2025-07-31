import base64
import json
import os
import re
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
# from openai import OpenAI
# import openai


import matplotlib
from tqdm import tqdm

#matplotlib.use('TkAgg')
# openai.api_key = "sk-OUx2ZUIE43zd4ar198189277E53a4c52829eFbE46e863680"
client = None
# client = OpenAI(
#     api_key="sk-OUx2ZUIE43zd4ar198189277E53a4c52829eFbE46e863680",  # 换成你的 key
#     base_url="https://api.gptapi.us/v1"
# )

ANNOTATION_ROOT = "./dataset/Annotation"
DATA_ROOT = "./dataset/all_data"
RESULT_ROOT = "./results"


class MLLM:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.prompt = "请查看这张图片，它被分成了 24 个编号区域（从左到右，从上到下）。请指出你认为图片中烟雾（smoke）出现在哪些编号区域？"

    def image_to_base64(self, img_array, format):
        success, encoded_img = cv2.imencode(format, img_array)
        if not success:
            raise RuntimeError('Failed to encode image')

        base64_bytes = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        return f'data:image/{format[1:]};base64,{base64_bytes}'

    def retrieve_grid_number(self, response_content):
        numbers = re.findall(r'\d+', response_content)
        numbers = [int(number) for number in numbers]
        return numbers

    def send_request(self, img, format='.jpg'):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.image_to_base64(img, format),
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        content = response.choices[0].message.content
        return self.retrieve_grid_number(content), content


class ImagePreprocess:

    @staticmethod
    def add_grid(image_path, rows=5, cols=5):
        img = cv2.imread(image_path)
        h, w, _ = img.shape

        cell_h = h // rows
        cell_w = w // cols

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 0, 255)

        region_id = 1
        for i in range(rows):
            for j in range(cols):
                x1, y1 = j * cell_w, i * cell_h
                x2, y2 = (j + 1) * cell_w, (i + 1) * cell_h

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

                cx = x1 + cell_w // 2
                cy = y1 + cell_h // 2

                label = str(region_id)
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy - text_size[1] // 2
                cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)
                region_id += 1
        return img

    @staticmethod
    def add_grid_to_patch(img, patch_box, rows=5, cols=5):
        w = abs(patch_box[2] - patch_box[0])
        h = abs(patch_box[3] - patch_box[1])

        cell_h = h // rows
        cell_w = w // cols

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(cell_w, cell_h) / 50.0
        thickness = 1
        color = (0, 0, 255)

        region_id = 1
        for i in range(rows):
            for j in range(cols):
                x1, y1 = j * cell_w + patch_box[0], i * cell_h + patch_box[1]
                x2, y2 = (j + 1) * cell_w + patch_box[0], (i + 1) * cell_h + patch_box[1]

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

                cx = x1 + cell_w // 2
                cy = y1 + cell_h // 2

                label = str(region_id)
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy - text_size[1] // 2
                cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)
                region_id += 1
        return img[patch_box[1], patch_box[3], patch_box[0]:patch_box[2]]

def get_annotation_grid_number(annotation_path, img, rows=5, cols=5):
    h, w, _ = img.shape
    cell_h, cell_w = h // rows, w // cols
    with open(annotation_path, 'r') as f:
        json_data = json.load(f)
        boxes = json_data['det_boxes']

        covered_grids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            start_row = y1 // cell_h
            end_row = y2 // cell_h
            start_col = x1 // cell_w
            end_col = x2 // cell_w
            for i in range(start_row, end_row + 1):
                for j in range(start_col, end_col + 1):
                    grid_number = i * cols + j + 1
                    covered_grids.append(grid_number)
    return covered_grids


def test():
    image_path = './dataset/day/20160604_FIRE_rm-n-mobo-c/1465065600_+00000.jpg'
    img_rgb = ImagePreprocess.add_grid(image_path)
    mllm = MLLM()
    mllm.send_request(img_rgb)
    plt.imshow(img_rgb)
    plt.show()



def compute_grid_IoU(prediction: set, annotation: set):
    intersection = prediction.intersection(annotation)
    union = prediction.union(annotation)
    IoU = len(intersection) / len(union)
    return IoU

def add_bbox(annotation_path, img, color=(0, 0, 255), thickness=2):
    with open(annotation_path, 'r') as f:
        json_data = json.load(f)
        boxes = json_data['det_boxes']
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def evaluate(image_path, annotation_path):
    img_rgb = ImagePreprocess.add_grid(image_path)
    annotation_grids = get_annotation_grid_number(annotation_path, img_rgb)
    mllm = MLLM.get_instance()
    predict_grids, content = mllm.send_request(img_rgb)
    iou = compute_grid_IoU(set(predict_grids), set(annotation_grids))
    img_with_box = add_bbox(annotation_path, img_rgb)
    return {
        'iou': iou,
        'predict_grids': predict_grids,
        'annotation_grids': annotation_grids,
        'img': img_with_box,
        'response': content
    }


def save_result(path: str, evaluation_result: dict, image_path: str):
    path = Path(os.path.join(path, image_path)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    img = evaluation_result.pop('img')
    json_path = path / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f)
    cv2.imwrite(str(path / 'img.jpg'), img)

def main():
    root_dir = Path(ANNOTATION_ROOT)
    all_ious = []
    for folder in tqdm(root_dir.iterdir()):
        if folder.is_dir():
            for label in folder.iterdir():
                if label.name.endswith('.json'):
                    image_name = label.name.replace('.json', '.jpg')
                    image_path = os.path.join(DATA_ROOT, folder.name, image_name)
                    annotation_path = str(label)
                    evaluation = evaluate(image_path, annotation_path)
                    iou = evaluation['iou']
                    all_ious.append(iou)
                    save_result(RESULT_ROOT, evaluation, os.path.join(folder.name, image_name))

if __name__ == "__main__":
    main()
