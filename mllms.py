import base64
import io

import cv2
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModel, LlavaProcessor, LlavaForConditionalGeneration, pipeline, \
    AutoModelForCausalLM, AutoTokenizer
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
            tokenizer="/d1/sunyu/sora/unified-io-2.pytorch/checkpoints/tokenizer.model",
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
        answer = self.runner.refexp(image, prompt)

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
        model_id = "Qwen/Qwen2.5-VL-32B-Instruct"  # 或 Qwen2.5-VL-32B-Instruct
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image, prompt):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
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
        return output_text[0]

