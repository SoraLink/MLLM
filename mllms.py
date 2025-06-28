import base64
import io

import cv2
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaProcessor, LlavaForConditionalGeneration


class MLLM_LLAVA:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.processor = LlavaProcessor.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            trust_remote_code=True
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            torch_dtype=torch.float16,
            device_map="sequential",
            trust_remote_code=True
        )
        self.prompt1 = (
            "<image>\n"
            'Detect all smoke and output bounding box like'
            '[[x1 y1 x2 y2], [x1 y1 x2 y2]]'
            "If you cannot find any smoke return 'None'"
        )

        self.prompt2 = (
            "<image>\n"
            "what is in the image"
            'Please look at this image, which is divided into 24 numbered regions '
            '(from left to right, top to bottom). '
            'Please output the numbered regions that contain smoke in JSON format as '
            'a list of dicts like [{"region": 1}, {"region": 2}].'
            # "请你只用一句话描述图片中是否有烟雾。如果有，出现在哪些编号区域？不要输出其他内容。"
        )

    def predict(self, image, is_grid=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        prompt = self.prompt2 if is_grid else self.prompt1
        inputs = self.processor(
            images=[image],
            text=[prompt],
            return_tensors="pt"
        ).to("cuda")

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_ids = outputs.sequences

        answer = self.processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        return answer


class InternVL3:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.prompt1 = (
            "<image>\n"
            'Detect all smoke and output bounding box like'
            '[[x1 y1 x2 y2], [x1 y1 x2 y2]]'
            "If you cannot find any smoke return empty list []"
        )

        self.prompt2 = (
            "<image>\n"
            "what is in the image"
            'Please look at this image, which is divided into 24 numbered regions '
            '(from left to right, top to bottom). '
            'Please output the numbered regions that contain smoke in JSON format as '
            'a list of dicts like [{"region": 1}, {"region": 2}].'
            "If you cannot find any smoke return empty list []"
        )
        self.processor = AutoProcessor.from_pretrained(
            "/path/to/InternVL3-Chat-13B", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "/path/to/InternVL3-Chat-13B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def predict(self, image, is_grid=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        prompt = self.prompt2 if is_grid else self.prompt1
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to("cuda")

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_ids = outputs.sequences
        answer = self.processor.tokenizer.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=True
        )
        print(answer)
        return answer