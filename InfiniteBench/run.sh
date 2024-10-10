module use /soft/modulefiles/
module load conda
conda activate Infinite_Bench


export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

model_name="meta-llama/Llama-3.1-8B"

python3 eval.py --model_name=$model_name --task="math_calc" --verbose
