

# MODEL_PATH="$1"
# set -x
# RESULT_PATH=$MODEL_PATH/results/
# mkdir -p $RESULT_PATH
# export PYTHONPATH=./accelerated-model-architectures

# /proj/checkpoints/bharat/hf_cache2/accelerate/default_config.yaml 


set -x
MODEL_PATH=$1
RESULT_PATH=$MODEL_PATH/results/
mkdir -p $RESULT_PATH
export HF_TOKEN=<put your token>
export PYTHONPATH=./accelerated-model-architectures:.
GLOBAL_MODEL_PARAMS="dtype=bfloat16,max_length=4096,use_cache=False"

# GLOBAL_MODEL_PARAMS="dtype=bfloat16,max_length=2048,use_cache=False"
# 
# GLOBAL_MODEL_PARAMS="dtype=bfloat16,max_length=1024,use_cache=False"

MODEL_ARGS="pretrained=${MODEL_PATH},$GLOBAL_MODEL_PARAMS"


accelerate launch -m lm_eval --batch_size 32 --model hf --model_args $MODEL_ARGS \
	--tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext,lambada_openai | tee $RESULT_PATH/results.log



# accelerate launch -m lm_eval --batch_size 32 --model hf --model_args $MODEL_ARGS \
# 	--task mmlu --num_fewshot 5 | tee $RESULT_PATH/mmlu.log
 