import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import base64
from io import BytesIO
from PIL import Image

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

MODEL_ID = "/data/input/amanr/olmOCR-7B-0225-preview"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True)

DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


def preprocess_and_tokenize(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    main_image = Image.open(BytesIO(base64.b64decode(encoded_image_text)))
    return processor(
        text=[text],
        images=[main_image],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head", "visual"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

oneshot(
    model=model,
    tokenizer=MODEL_ID,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
)


SAVE_DIR = MODEL_ID + "-fp8-kv"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)