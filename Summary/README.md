# Evaluating LLMs on summary tasks(XSUM, CNN/Daily Mail)

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

## Running on Summary tasks

Step1：Generate Datasets for tasks (XSUM, CNN/Daily Mail)
```bash
python get_data.py --dataset cnn_dailymail #{xsum,multi_news,cnndm}
```

Step2: Generate the outputs
```bash
GPU=0,1,2,3
task=cnndm 
shot=0
model=huggyllama/llama-7b
model_arch=gpt_neox   # llama / gpt_neox / opt 
CUDA_VISIBLE_DEVICES=${GPU} python -u run_helm.py \
  --input_path data/${task}_${shot}shot.jsonl \
  --output_path ${task}-${shots}-${model_arch}.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
```

Step3: Evaluate the outputs
```bash
model_arch=llama
shots=0
python eval_helm.py \
  --task ${task} \
  --model ${model_arch} \
  --input_path ${task}-${shots}-${model_arch}.jsonl\
  --output_path ${task}-${shots}-${model_arch}-rouge.txt
```
