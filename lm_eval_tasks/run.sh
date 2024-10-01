module use /soft/modulefiles/
module load conda

conda activate lm_eval

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

python evaluator.py --model-name="meta-llama/Meta-Llama-3-8B" --task-name="openbookqa" --num-fewshot=0

# task=openbookqa #mathqa,boolq,copa,winogrande
# shots=0 
# python -u generate_task_data.py \
#   --output-file ${task}-${shots}.jsonl \
#   --task-name ${task} \
#   --num-fewshot ${shots}

# GPU=0
# model=huggyllama/llama-7b
# model_arch=llama   # llama / gpt-neox / opt
# CUDA_VISIBLE_DEVICES=${GPU} python -u run_lm_eval_harness.py \
#   --input-path ${task}-${shots}.jsonl \
#   --output-path ${task}-${shots}-${model_arch}.jsonl \
#   --model-name ${model} \
#   --model-type ${model_arch}

# model_arch=llama   # llama / gpt_neox / opt 
# python -u evaluate_task_result.py \
#   --result-file ${task}-${shots}-${model_arch}.jsonl \
#   --task-name ${task} \
#   --num-fewshot ${shots} \
#   --model-type ${model_arch} \
#   --ret-path ${task}-${shots}-${model_arch}.txt \
#   --output-path ${task}-${shots}-${model_arch}.jsonl



# from lm_eval import models
# from lm_eval import tasks
# from lm_eval import evaluate