import torch
from torch import nn
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from models.base_model import BaseModel

class PixtralModel(BaseModel):
    def __init__(self, model_id: str, cache_dir:str):
        super().__init__()
        
        self.model     =  LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir = cache_dir, device_map="auto", torch_dtype=torch.bfloat16) 
        self.processor =  AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
    
    def get_tokenizer(self):
        return self.processor.tokenizer

    def generate_output(self, batch, max_new_tokens):
        return self.generate(batch, self.model, self.processor, max_new_tokens=max_new_tokens)
