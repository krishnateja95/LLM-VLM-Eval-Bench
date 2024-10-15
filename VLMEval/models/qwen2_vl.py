import torch
from torch import nn
from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from models.base_model import BaseModel

class MLLaMA_model(BaseModel):
    def __init__(self, model_id: str, cache_dir:str):
        super().__init__()

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    def get_tokenizer(self):
        return self.processor.tokenizer

    def generate_output(self, batch, max_new_tokens=30):
        return self.generate(self, batch, self.model, self.processor, max_new_tokens=max_new_tokens)
