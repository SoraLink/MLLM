import base64
import io

import cv2
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, AutoModel, LlavaProcessor, LlavaForConditionalGeneration, pipeline


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
        print("\n===== DEBUG: Image Info =====")
        print(f"is_grid = {is_grid}")
        print(f"Image type: {type(image)}")  # 应该是 <class 'PIL.Image.Image'>
        print(f"Image mode: {image.mode}")   # 应该是 'RGB'
        print(f"Image size: {image.size}")   # 应该是 (width, height)，如 (1024, 1024)
        print("==========================\n")
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
            "If you cannot find any smoke return empty list [] without other words"
        )

        self.prompt2 = (
            "<image>\n"
            "what is in the image"
            'Please look at this image, which is divided into several numbered regions '
            '(from left to right, top to bottom). '
            'Please output the numbered regions that contain smoke in JSON format as '
            'a list of dicts like [{"region": 1}, {"region": 2}].'
            "If you cannot find any smoke return empty list [] without other words"
        )
        #from transformers import pipeline, AutoProcessor, AutoModel, AutoTokenizer
        #import torch

        # 1. 先加载 processor 和模型
       # tokenizer = AutoTokenizer.from_pretrained("/d1/sunyu/sora/MLLM/InternVL3-38B-hf", trust_remote_code=True)
       # processor = AutoProcessor.from_pretrained(
       #             "/d1/sunyu/sora/MLLM/InternVL3-38B-hf", trust_remote_code=True
       #             )
       # model = AutoModel.from_pretrained(
       #         "/d1/sunyu/sora/MLLM/InternVL3-38B-hf",
       #         torch_dtype=torch.float16,
       #         device_map="balanced"
       #         )

        # 2. 子类化原 pipeline
        #BaseClass = type(pipeline("image-text-to-text"))
        #class MyImageTextToText(BaseClass):
#            def preprocess(self, inputs):
#                data = super().preprocess(inputs)
                                    # 强制把图像 tensor 转成 float16
#               if "pixel_values" in data:
#                    data["pixel_values"] = data["pixel_values"].to(torch.float16)
#                return data

                                                                # 3. 用你的子类去实例化 pipeline
        #self.pipe = MyImageTextToText(model=model, tokenizer=tokenizer, feature_extractor=processor, device="cuda")
        self.pipe = pipeline(
            "image-text-to-text",
            model="OpenGVLab/InternVL3-14B-hf",  # 或者 "OpenGVLab/InternVL3-14B"、"OpenGVLab/InternVL3-2B" 等
            device_map="auto",               # 自动分配到你可见的 GPU
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    def predict(self, image, is_grid=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        #out = self.processor.image_processor(images=image, return_tensors="pt")
        #print("DEBUG dtype before patch:", out["pixel_values"].dtype)
        #image = image.resize((448, 448), Image.BILINEAR)
        print("\n===== DEBUG: Image Info =====")
        print(f"is_grid = {is_grid}")
        print(f"Image type: {type(image)}")  # 应该是 <class 'PIL.Image.Image'>
        print(f"Image mode: {image.mode}")   # 应该是 'RGB'
        print(f"Image size: {image.size}")   # 应该是 (width, height)，如 (1024, 1024)
        print("==========================\n")
        prompt = self.prompt2 if is_grid else self.prompt1
        print(prompt)
        print(prompt.count("<image>"))
        #print(self.processor.image_placeholder)  # 检查占位符是否是 `<image>`
        #print(self.processor.num_images)        # 检查默认期望的图像数量
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        res = self.pipe(text=messages, return_full_text=False)
        answer = res[0]["generated_text"]
        print(answer)
        return answer
