import base64
import io

import cv2
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, AutoModel, LlavaProcessor, LlavaForConditionalGeneration, pipeline, \
    AutoModelForCausalLM

from uio2.model import UnifiedIOModel
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.preprocessing import build_batch
from uio2.prompt import Prompt
from uio2.runner import TaskRunner


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
            'Detect all smoke and output bounding box like'
            '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
            "If you cannot find any smoke return empty list [] without other words"
        )

        self.prompt2 = (
            "what is in the image"
            'Please look at this image, which is divided into several numbered regions '
            '(from left to right, top to bottom). '
            'Please output the numbered regions that contain smoke in JSON format as '
            'a list of dicts like [{"region": 1}, {"region": 2}].'
            "If you cannot find any smoke return empty list [] without other words"
        )
        self.pipe = pipeline(
            "image-text-to-text",
            model="/d1/sunyu/sora/MLLM/InternVL3-78B-hf",
            device_map="balanced",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    def predict(self, image, is_grid=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        prompt = self.prompt2 if is_grid else self.prompt1
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        res = self.pipe(messages, return_full_text=False)
        answer = res[0]["generated_text"]
        print("prompt: ", prompt)
        print("response: ", answer)
        return answer


class UIO2:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.preprocessor = UnifiedIOPreprocessor.from_pretrained(
            "allenai/uio2-preprocessor",
            tokenizer="/d1/sunyu/sora/unified-io-2.pytorch/checkpoints/tokenizer.model",
            #trust_remote_code=True
        )
        self.model = UnifiedIOModel.from_pretrained(
            "allenai/uio2-xxl",
            #torch_dtype=torch.float32,
            #device_map="sequential",
            #trust_remote_code=True
        )
        prompts = Prompt(
            original_flag=False,
            manual_flag=True,
            gpt3_flag=False,
            single_prompt=True)
        self.runner = TaskRunner(self.model, self.preprocessor, prompts=prompts)
        self.prompt1 = (
            "smoke"
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
        if is_grid:
            answer = self.runner.vqa(image, self.prompt2)
        else:
            answer = self.runner.refexp(image, self.prompt1)

        print(answer)

        return str([answer])
