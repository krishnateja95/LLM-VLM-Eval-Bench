

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.LLaVA import LlavaModel
import argparse

from metrics import AverageMeter, calculate_bleu_from_raw_data, calculate_rouge_from_raw_data
from datasets.cococap2017 import CoCoCaption2017

def custom_collate_fn(batch):
    ids = [item['ID'] for item in batch]
    titles = [item['title'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    image_urls = [item['image_url'] for item in batch]
    entities = [item['entities'] for item in batch]
    
    return {
        'ID': ids,
        'title': titles,
        'question': questions,
        'answer': answers,
        'image_url': image_urls,
        'entities': entities
    }

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmark VLMs.')
    parser.add_argument('--model_id', type=str, default = "llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--dataset_name', type=str, default = "lmms-lab/COCO-Caption2017")
    parser.add_argument('--cache_dir', type=str, default = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
    args = parser.parse_args()

    dataset = CoCoCaption2017(args.dataset_name, args.cache_dir)

    test_dataloader = dataset.get_test_dataloader()
    validation_dataloader = dataset.get_val_dataloader()

    # train_dataloader = DataLoader(dataset['train'], batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    # validation_dataloader = DataLoader(dataset['validation'], batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    # test_dataloader = DataLoader(dataset['test'], batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = LlavaModel(model_id = args.model_id, cache_dir = args.cache_dir)

    rouge_1_score_avg = AverageMeter()
    rouge_2_score_avg = AverageMeter()
    rouge_L_score_avg = AverageMeter()
    bleu_score_avg    = AverageMeter()

    for batch in validation_dataloader:
        
        max_new_tokens = []
        for answer in batch['answer']:
            max_new_tokens.append(len(model.get_tokenizer().tokenize(answer)))

        preds = model.generate_output(batch=batch, max_new_tokens=max_new_tokens)

        rouge_scores = calculate_rouge_from_raw_data(preds, batch['answer'], model.get_tokenizer())
        print(f"ROUGE-1: {rouge_scores['ROUGE-1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['ROUGE-2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['ROUGE-L']:.4f}")

        bleu_score = calculate_bleu_from_raw_data(preds, batch['answer'], model.get_tokenizer())
        print(f"BLEU Score: {bleu_score:.4f}")


        rouge_1_score_avg.update(rouge_scores['ROUGE-1'], len(preds))
        rouge_2_score_avg.update(rouge_scores['ROUGE-2'], len(preds))
        rouge_L_score_avg.update(rouge_scores['ROUGE-L'], len(preds))
        bleu_score_avg.update(bleu_score, len(preds))

        break 

    print("Rouge-1 Score", rouge_1_score_avg.avg)
    print("Rouge-2 Score", rouge_2_score_avg.avg)
    print("Rouge-L Score", rouge_L_score_avg.avg)
    print("BLEU Score", bleu_score_avg.avg)

    