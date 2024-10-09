import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from vllm import LLM, SamplingParams

from argparse import ArgumentParser, Namespace
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS


MAX_POSITION_ID = 200000
TRUNCATE_LEN = 200000

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(model, tok: AutoTokenizer, input_text: str, max_tokens: int, verbose: bool = False) -> str:
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    
    outputs = model.generate([input_text], sampling_params)
    
    output = outputs[0].outputs[0].text
    print("Chunked generation:", output)
    return output


def load_model(model_name, ngpu = 8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    start_time = time.time()
    llm = LLM(model=model_name, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tokenizer


if __name__ == "__main__":
    
    p = ArgumentParser()
    p.add_argument("--task", type=str, required=True, help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.")
    p.add_argument('--data_dir',type=str, default='../data', help="The directory of data.")
    p.add_argument("--output_dir", type=str, default="../results", help="Where to dump the prediction results.") 
    p.add_argument("--model_path", type=str, help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.")
    p.add_argument("--model_name", type=str, default="gpt4", help="For `compute_scores.py` only, specify which model you want to compute the score for.")
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    args =  p.parse_args()

    
    model_name = args.model_name

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)
    
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (result_dir / f"preds_{data_name}.jsonl")
    else:
        output_path = (result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl")

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose)
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)