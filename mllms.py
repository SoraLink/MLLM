import base64
import io
import re

import cv2
import torch
from PIL import Image
from accelerate import dispatch_model, infer_auto_device_map
from huggingface_hub import InferenceClient
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModel, LlavaProcessor, LlavaForConditionalGeneration, pipeline, \
    AutoModelForCausalLM, AutoTokenizer, AutoModelForZeroShotObjectDetection, Idefics2ForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


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

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        prompt = '<image>\n' + prompt
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

        generated_ids = outputs.sequences[0]
        new_ids = generated_ids[prompt_len:]

        answer = self.processor.tokenizer.decode(
            new_ids,
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
        self.pipe = pipeline(
            "image-text-to-text",
            model="OpenGVLab/InternVL3-14B-hf",
            device_map="balanced",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
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
            tokenizer="/data/sora/unified-io-2.pytorch/checkpoints/tokenizer.model",
        )
        self.model = UnifiedIOModel.from_pretrained(
            "allenai/uio2-xxl",
        )
        prompts = Prompt(
            original_flag=False,
            manual_flag=True,
            gpt3_flag=False,
            single_prompt=True)
        self.runner = TaskRunner(self.model, self.preprocessor, prompts=prompts)

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        answer = self.runner.refexp(image, "the smoke")

        print(answer)

        return str([answer])

class QwenVL:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_id = ("Qwen/Qwen2.5-VL-7B-Instruct")  # 或 Qwen2.5-VL-32B-Instruct
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((448, 448))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(output_text[0])
        return self.extract_json_from_markdown(output_text[0])

    def batch_predict(self, images, prompt):

        texts = []
        images_inputs_batch = []
        videos_inputs_batch = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((448, 448))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(text)
            images_inputs_batch.append(image_inputs)
            videos_inputs_batch.append(video_inputs)

        inputs = self.processor(
            text=texts,
            images=images_inputs_batch,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        outputs = []
        for output in output_text:
            outputs.append(self.extract_json_from_markdown(output))
        return outputs

    def extract_json_from_markdown(self, text: str) -> str:
        """
        从包含 ```json ... ``` 的字符串中提取纯 JSON 字符串。
        如果没有包裹则原样返回。
        """
        # 使用正则提取 ```json ... ``` 包裹的内容
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


class GroundingDINO:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=True,
            trust_remote_code=True  # GroundingDINO 通常需要
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            token=True,
            trust_remote_code=True  # GroundingDINO 通常需要
        ).to('cuda')

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        text = "smoke."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            target_sizes=[image.size[::-1]]  # 注意尺寸顺序翻转
        )
        print(len(results))
        result = results[0]
        answer = result['boxes']
        print(str(answer.tolist()))
        return str(answer.tolist())

class IDEFICS2:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_id = "HuggingFaceM4/idefics2-8b"

        # 1) 加载模型到 meta，节省内存，后续再分配设备
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # 或者 torch.float16 视GPU而定
            low_cpu_mem_usage=True,
            device_map=None  # 先不要auto
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 2) 自动推断分片，但不要把视觉分支拆开
        #    no_split 避免层被切碎；include_buffers=True 确保 buffer 也随模块走同一设备
        max_mem = {i: "75GiB" for i in range(torch.cuda.device_count())}  # 根据你的机器改
        device_map = infer_auto_device_map(
            self.model,
            dtype=torch.bfloat16,
            max_memory=max_mem,
            no_split_module_classes=[
                "Idefics2VisionEncoderLayer",
                "Idefics2DecoderLayer",
                "Idefics2VisionModel"
            ],
            offload_buffers=True
        )

        # 3) 强制把整个视觉分支放到同一张GPU（选第一张可用卡）
        #    这样就不会在 vision embeddings 里出现 bucketize 的跨设备冲突
        vision_gpu = next(iter(max_mem.keys()))  # 不硬编码为0，自动取一张
        for name in list(device_map.keys()):
            if name.startswith("model.vision_model"):
                device_map[name] = vision_gpu

        # 4) 按映射把模型分发到多卡
        dispatch_model(self.model, device_map=device_map, offload_dir=None)

        # 5) pipeline 直接用已经分片的 model；不要再传 device 或 device_map
        self.pipe = pipeline(
            task="image-text-to-text",
            model=self.model,
            processor=self.processor,
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor
            # 不要再传 device / device_map，避免覆盖我们上面的分片
        )

    def predict(self, image, prompt):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)

        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt

        result = self.pipe({
            "text": prompt,
            "images": pil_img
        })
        return result[0]["generated_text"]



