# Evaluating LLMs on LM-Eval Tasks

## Setup

```bash
conda create -n lm_eval python=3.10
conda activate lm_eval

pip install crfm_helm==0.2.3
pip install lm_eval==0.3.0
pip install requests
pip install accelerate
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.28.1
```

## Running on QA tasks

Step1: Download the Dataset
```bash
task=openbookqa #mathqa,boolq,copa,winogrande
shots=0 
python -u generate_task_data.py \
  --output-file ${task}-${shots}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}
```

Step2: Generate Outputs on the downloaded datasets 
```bash
GPU=0
model=huggyllama/llama-7b
model_arch=llama   # llama / gpt-neox / opt
CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}
```

Step3: Evaluate the result
```bash
model_arch=llama   # llama / gpt_neox / opt 
python -u evaluate_task_result.py \
  --result-file ${task}-${shots}-${model_arch}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch} \
  --ret-path ${task}-${shots}-${model_arch}.txt
```