# Evaluating LLMs on Generation tasks

## Setup
```bash
conda create -n summary_tasks python=3.10
conda activate summary_tasks

pip install crfm_helm==0.2.3
pip install lm_eval==0.3.0
pip install requests
pip install accelerate
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.28.1
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
