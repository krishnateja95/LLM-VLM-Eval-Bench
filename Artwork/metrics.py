

import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from collections import Counter
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def calculate_cider(labels, logits, tokenizer):
    
    references = [tokenizer.tokenize(label) for label in labels]
    candidates = [tokenizer.tokenize(logit) for logit in logits]

    # Compute term frequency (TF)
    tf = Counter()
    for ref in references:
        for word in ref:
            tf[word] += 1
    
    # Compute inverse document frequency (IDF)
    idf = Counter()
    for ref in references:
        unique_words = set(ref)
        for word in unique_words:
            idf[word] += 1
    
    idf = {word: np.log(len(references) / (1 + doc_count)) for word, doc_count in idf.items()}
    scores = []
    
    for candidate in candidates:
        # Get candidate word frequency
        candidate_counter = Counter(candidate)
        score = 0
        for word in candidate_counter:
            tf = candidate_counter[word] / len(candidate)
            idf_score = idf.get(word, 0)
            score += tf * idf_score
        scores.append(score)
    
    return np.mean(scores)





def calculate_bleu_from_raw_data(logits_list, labels_list, tokenizer):
    
    bleu_scores = []
    
    # Iterate through each pair of logits (model output) and labels (ground truth)
    for logits, label in zip(logits_list, labels_list):
        # Tokenize both model outputs and ground truth labels
        tokenized_output = tokenizer.tokenize(logits)
        tokenized_label = tokenizer.tokenize(label)
        
        # BLEU expects the reference to be a list of lists of tokens
        reference = [tokenized_label]
        candidate = tokenized_output
        
        # Calculate BLEU score for this example
        bleu_score = sentence_bleu(reference, candidate)
        
        # Append BLEU score to the list
        bleu_scores.append(bleu_score)
    
    # Calculate average BLEU score over the batch
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    
    return avg_bleu_score



def calculate_rouge_from_raw_data(logits_list, labels_list, tokenizer):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Store the scores for each example
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # Iterate through each pair of logits (model output) and labels (ground truth)
    for logits, label in zip(logits_list, labels_list):
        # Tokenize both model outputs and ground truth labels
        tokenized_output = tokenizer.decode(tokenizer(logits, return_tensors="pt").input_ids.squeeze(), skip_special_tokens=True)
        tokenized_label = tokenizer.decode(tokenizer(label, return_tensors="pt").input_ids.squeeze(), skip_special_tokens=True)

        # Compute ROUGE scores for this example
        scores = scorer.score(tokenized_label, tokenized_output)
        
        # Append individual ROUGE scores
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # Calculate average ROUGE scores over all examples
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    return {
        'ROUGE-1': avg_rouge1,
        'ROUGE-2': avg_rouge2,
        'ROUGE-L': avg_rougeL
    }




def calculate_perplexity_from_raw_data(logits_list, labels_list, tokenizer):
    
    tokenized_logits = []
    tokenized_labels = []
    
    # Tokenize both logits (model outputs) and labels (ground truth)
    for logits, label in zip(logits_list, labels_list):
        # Tokenize logits (model output) and ground truth labels
        tokenized_output = tokenizer(logits, return_tensors="pt").input_ids  # shape: (1, seq_len)
        tokenized_label = tokenizer(label, return_tensors="pt").input_ids  # shape: (1, seq_len)
        
        # Append to the lists
        tokenized_logits.append(tokenized_output)
        tokenized_labels.append(tokenized_label)

    # Convert the list of tensors to a single tensor
    logits = torch.cat(tokenized_logits, dim=0)  # Shape: (total_batch_size, seq_len)
    labels = torch.cat(tokenized_labels, dim=0)  # Shape: (total_batch_size, seq_len)
    
    # Make sure the logits have the correct vocab size, vocab_size should be > 1
    vocab_size = tokenizer.vocab_size
    logits = torch.randn(logits.size(0), logits.size(1), vocab_size)  # random logits (replace with actual logits if available)
    
    # Flatten logits and labels
    logits = logits.view(-1, vocab_size)  # (total_tokens, vocab_size)
    labels = labels.view(-1)  # (total_tokens,)
    
    # Compute log softmax of logits
    log_probs = F.log_softmax(logits, dim=-1)  # (total_tokens, vocab_size)
    
    # Compute negative log likelihood loss
    nll_loss = F.nll_loss(log_probs, labels, reduction='mean')  # mean loss over all tokens
    
    # Perplexity is the exponential of the NLL loss
    perplexity = torch.exp(nll_loss)
    
    return perplexity.item()




if __name__ == '__main__':

    logits_list = ["Hello, how are you?", "What is the weather today in the morning?"]  # non-tokenized model outputs
    labels_list = ["Hello, how are he?", "What is the dog today?"]  # non-tokenized ground truth labels

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # perplexity = calculate_perplexity_from_raw_data(logits_list, labels_list, tokenizer)
    # print(f"Perplexity: {perplexity}")

    rouge_scores = calculate_rouge_from_raw_data(logits_list, labels_list, tokenizer)
    print(f"ROUGE-1: {rouge_scores['ROUGE-1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['ROUGE-2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['ROUGE-L']:.4f}")

    bleu_score = calculate_bleu_from_raw_data(logits_list, labels_list, tokenizer)
    print(f"BLEU Score: {bleu_score:.4f}")

    cider_score = calculate_cider(labels=labels_list, logits=logits_list, tokenizer=tokenizer)
    print(f"CIDEr Score: {cider_score:.4f}")