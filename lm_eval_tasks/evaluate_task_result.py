import argparse
import json
import os
from lm_eval import evaluator, tasks

from functools import partial
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer


from itertools import zip_longest
import numpy as np


def grouper(n, iterable, fillvalue):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def shrink_seq(examples, min_seq=None):
    length = examples["obs"].shape[-1]

    new_length = length // 2

    if min_seq is not None:
        if new_length < min_seq:
            return examples

    max_length = np.max(examples["eval_mask"] * np.arange(0, length)) + 1

    if max_length < new_length:
        examples["obs"] = examples["obs"][:, :new_length]
        examples["target"] = examples["target"][:, :new_length]
        examples["eval_mask"] = examples["eval_mask"][:, :new_length]

        return shrink_seq(examples, min_seq=min_seq)
    else:
        return examples


def sample_batch(examples, bs, zero_example_shape):
    zero_example = {
        "obs": np.zeros_like(zero_example_shape["obs"]),
        "target": np.zeros_like(zero_example_shape["target"]),
        "eval_mask": np.zeros_like(zero_example_shape["eval_mask"]),
        "ctx_length": 0,
    }

    for batch in grouper(bs, examples, zero_example):
        batch_flattened = {
            "obs": [],
            "target": [],
            "eval_mask": [],
            "ctx_length": [],
            "text": [],
        }

        for sample in batch:
            batch_flattened["obs"].append(sample["obs"])
            batch_flattened["target"].append(sample["target"])
            batch_flattened["eval_mask"].append(sample["eval_mask"])
            batch_flattened["ctx_length"].append(sample["ctx_length"])
            batch_flattened["text"].append(sample["text"])

        batch_flattened["obs"] = np.array(batch_flattened["obs"])
        batch_flattened["target"] = np.array(batch_flattened["target"])
        batch_flattened["eval_mask"] = np.array(batch_flattened["eval_mask"])
        batch_flattened["ctx_length"] = np.array(batch_flattened["ctx_length"])

        yield batch_flattened


def json_to_key(obj):
    return json.dumps(obj)

def process_init():
    global tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    if model_name == "EleutherAI/gpt-neox-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    elif model_name == "meta-llama/Llama-2-7b-hf":
        tokenizer =  LlamaTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_bos_token = False

def process_request(x, seq):
    global tokenizer

    ctx, cont = x
#     ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
        "prompt": ctx_text,
        "target": cont_text,
        "text": all_text,
    }

class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(processes=1, initializer=process_init)
        # self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output





if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='result.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--start-ratio', type=float, default=0.1)
    parser.add_argument('--recent-ratio', type=float, default=0.1)
    parser.add_argument('--ret-path', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    args = parser.parse_args()
    
    if args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "meta-llama/Llama-2-7b-hf"
    else:
        assert False

    seq = 1024
    total_batch = 1
    pe = 'fixed'
    class RealRunner:
        
        def __init__(self, args):
            
            self.results = {}
            
            with open(args.result_file, 'r') as f:
                
                for line in f:
                    if line.strip() == '':
                        continue
                    
                    item = json.loads(line)
                    
                    request = item['request']
                    result = item['result']
                    
                    self.results[json_to_key(request)] = result
            
            print(f"{len(self.results)} items in the cache")
        
        def eval(self, batch):
            
            from tasks.eval_harness import tokenizer
            
            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch['text']):
                
                request = {
                        "best_of": 1, 
                        "echo": True, 
                        "logprobs": 1, 
                        "max_tokens": 0, 
                        "model": "x", 
                        "n": 1, 
                        "prompt": text, 
                        "request_type": "language-model-inference", 
                        "stop": None, 
                        "temperature": 0, 
                        "top_p": 1
                    }
                
                key = json_to_key(request)
                
                correct = True
                
                if key in self.results:
                    result = self.results[key]
                    
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    obs = batch['obs'][i]
                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            
                            if i+1 >= len(tokens):
                                break
                            
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    avg_logprob = sum_lobprob / n_positive
                    
                    mask_loss.append( - avg_logprob)
            
                    each_correct.append( correct )
                    
                else:
                    assert False
                

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            
            
            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    results = evaluator.evaluate(adaptor, tasks.get_task_dict([args.task_name
                                                               #"lambada_openai",
                                                               #"piqa",
                                                               #"hellaswag",
                                                               #"winogrande",
                                                               #"mathqa",
                                                               #"pubmedqa",
                                                               # "boolq",
                                                               # "cb",
                                                               # "copa",
                                                               # "multirc",
                                                               # "record",
                                                               # "wic",
                                                               # "wsc",
                                                               ]), False, args.num_fewshot, None)
    
    dumped = json.dumps(results, indent=2)
    print(dumped)
    ret_path=args.ret_path
    file=open(ret_path,'a')
    file.write("method : "+ str(args.method) +"\n")
    file.write("task : "+ str(args.task_name) +"\n")
    file.write("start-ratio : " + str(args.start_ratio) + '\n')
    file.write("recent-ratio : " + str(args.recent_ratio) + '\n')
    file.write(dumped)
    file.write('\n*******************************************\n')
    