# Evaluating LLMs on Generation tasks

## Setup
```bash
conda create -n txt_gen python=3.11 -y
conda activate txt_gen

pip install -r requirements.txt
```

Evaluate LLM on wikitext dataset

```bash
model_name=huggyllama/llama-7b
task=wikitext # {wikitext, pg19}
python  eval_long_ppl.py \
  --model_name_or_path ${model_name} \
  --dataset_name ${task} \
  --num_samples  100 \
  --output_dir dense-ppl.txt
```
