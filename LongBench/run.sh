module use /soft/modulefiles/
module load conda
conda activate LongBench

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

model=llama2-7b-chat-4k

CUDA_VISIBLE_DEVICES=0,1,2,3 python pred.py --model $model --e
python eval.py --model $model --e
