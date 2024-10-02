import torch
from tqdm import tqdm
import os
import os.path as osp
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import argparse
import ssl
import urllib.request
import json
import sys

if __name__ == '__main__':

    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/llama/llama-7b")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_samples",type=int,default=1,)
    parser.add_argument("--output_dir",type=str,default="outputs/debug",)
    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")
    parser.add_argument("--num_eval_tokens", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    data = load_dataset(args.dataset_name, args.task, split=args.split)

    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=cache_dir, device_map="auto",
                                                 torch_dtype=torch.float16, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    kv_cache        = None

    num_eval_tokens = 0
    for text in data["text"]:
        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)

                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)

                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
            
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    ppl = torch.exp(torch.stack(nlls).mean())

    with open(args.output_dir, "w") as f:
        f.write(f"model name :{args.model_name_or_path}\n")
        f.write(f"mean ppl :{ppl.item()}\n")
        f.write(f"****************************************************\n")
    print(f"mean ppl {ppl}")