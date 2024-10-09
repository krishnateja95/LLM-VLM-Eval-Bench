import os
import json
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_evaluation():
    
    cache_dir = "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/"
    model_name = "huggyllama/llama-7b" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir = cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir= cache_dir, torch_dtype = torch.float16, device_map = "auto")

    result = evaluate(
            HFLM(
                pretrained=model,
                tokenizer=tokenizer, 
                batch_size=1, 
                max_length=None),
            get_task_dict(["mathqa"]),
            limit = None,
        )

    for task, res in result["results"].items():
        print(f"{task}: {res}")



if __name__ == '__main__':
    run_evaluation()

