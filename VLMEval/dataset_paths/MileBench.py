
from datasets import load_dataset
from torch.utils.data import DataLoader
# from metrics import AverageMeter, calculate_bleu_from_raw_data, calculate_rouge_from_raw_data

def custom_collate_fn(batch):
    question_id = [item['question_id'] for item in batch]
    question = [item['question'] for item in batch]
    answer = [item['answer'][0] for item in batch]
    image_url = [item['coco_url'] for item in batch]
    height = [item['height'] for item in batch]
    width = [item['width'] for item in batch]

    return {
        'question_id': question_id,
        'question': question,
        'answer': answer,
        'image_url': image_url,
        'height': height,
        'width':width
    }

class MileBench():
    def __init__(self, cache_dir):
        super().__init__()
        
        self.dataset_name = "FreedomIntelligence/MileBench"

        self.dataset_name = "HuggingFaceM4/TextCaps"

        self.cache_dir    = cache_dir
        self.dataset = load_dataset(self.dataset_name, cache_dir = self.cache_dir)

    def get_train_dataloader(self, batch_size = 32):
        return DataLoader(self.dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        
    def get_val_dataloader(self, batch_size = 32):
        return DataLoader(self.dataset['val'], batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
    def get_test_dataloader(self, batch_size = 32):
        return DataLoader(self.dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    def postprocess(self, pred_list):
        return [entry.split("ASSISTANT: ", 1)[1] for entry in pred_list]

    # def evaluate(self, model, dataset):
    #     rouge_1_score_avg = AverageMeter()
    #     rouge_2_score_avg = AverageMeter()
    #     rouge_L_score_avg = AverageMeter()
    #     bleu_score_avg    = AverageMeter()
        
    #     for batch in dataset:
            
    #         max_new_tokens = []
    #         for answer in batch['answer']:
    #             max_new_tokens.append(len(model.get_tokenizer().tokenize(answer)))

    #         preds = model.generate_output(batch=batch, max_new_tokens=max(max_new_tokens))
    #         preds = self.postprocess(preds)

    #         print("answer", batch['answer'])
    #         print()
    #         print("preds", preds)
    #         print()

    #         rouge_scores = calculate_rouge_from_raw_data(preds, batch['answer'], model.get_tokenizer())
    #         # print(f"ROUGE-1: {rouge_scores['ROUGE-1']:.4f}")
    #         # print(f"ROUGE-2: {rouge_scores['ROUGE-2']:.4f}")
    #         # print(f"ROUGE-L: {rouge_scores['ROUGE-L']:.4f}")

    #         bleu_score = calculate_bleu_from_raw_data(preds, batch['answer'], model.get_tokenizer())
    #         # print(f"BLEU Score: {bleu_score:.4f}")

    #         rouge_1_score_avg.update(rouge_scores['ROUGE-1'], len(preds))
    #         rouge_2_score_avg.update(rouge_scores['ROUGE-2'], len(preds))
    #         rouge_L_score_avg.update(rouge_scores['ROUGE-L'], len(preds))
    #         bleu_score_avg.update(bleu_score, len(preds))

    #     print("Rouge-1 Score", rouge_1_score_avg.avg)
    #     print("Rouge-2 Score", rouge_2_score_avg.avg)
    #     print("Rouge-L Score", rouge_L_score_avg.avg)
    #     print("BLEU Score", bleu_score_avg.avg)

        

if __name__ == '__main__':

    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
    
    dataset = MileBench(cache_dir)

    print("dataset", dataset)
        


