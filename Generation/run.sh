module use /soft/modulefiles/
module load conda
conda activate txt_gen

export HF_HOME='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
export HF_DATASETS_CACHE='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'


model_name=huggyllama/llama-7b
task=wikitext # {wikitext, pg19}
python  eval_long_ppl.py \
  --model_name_or_path ${model_name} \
  --dataset_name ${task} \
  --output_dir dense-ppl.txt