module use /soft/modulefiles/
module load conda

conda activate VLMEval

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

data="MathVista_MINI"
model=Llama-3.2-11B-Vision-Instruct

python3 run.py --data=$data --model=$model --verbose


