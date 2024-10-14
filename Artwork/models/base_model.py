import torch
from torch import nn
from PIL import Image
import requests
import io

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def generate(self, batch, model, processor, max_new_tokens):
        
        images = []
        for input_image_url in batch["image_url"]:
            image = Image.open(requests.get(input_image_url, stream=True).raw)
            images.append(image)
            print(image)
            exit()

        prompts = []
        for text in batch["question"]:
            conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": text},
                                ],
                            },
                        ]
            prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True)) 

        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to("cuda:0", torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        preds = processor.batch_decode(generate_ids, skip_special_tokens=True)
        
        return preds
    
