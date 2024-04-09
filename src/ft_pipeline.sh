#!/bin/bash

TRAIN=true
ITER=release

while [[ $# -gt 0 ]]; do
    case "$1" in
        --infer)
            TRAIN=false
            ;;
        *)
            # Assuming the first three arguments are positional
            if [[ -z $LOAD_CP ]]; then
                LOAD_CP="$1"
            elif [[ -z $SAVE_CP ]]; then
                SAVE_CP="$1"
            elif [[ $ITER = "release" ]]; then
                ITER="$1"
            fi
            ;;
    esac
    shift
done

source ~/.bashrc
CONDA_ENV_PATH='/home/cse/btech/cs1200389/.conda/envs/matllama2.0'
conda activate ${CONDA_ENV_PATH}

module load compiler/gcc/11.2.0
module unload compiler/cuda/11.0/compilervars
module load compiler/cuda/12.3/compilervars

if $TRAIN; then
    if [ "$ITER" = "release" ]; then
        python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type llama2 --load_dir $LOAD_CP --save_dir $SAVE_CP/$ITER/added-vocab --true_vocab_size 32007
    else
        python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type llama2 --load_dir $LOAD_CP --save_dir $SAVE_CP/$ITER/added-vocab --true_vocab_size 32007 --load_iters $ITER
    fi
	sh ./finetune.sh $SAVE_CP/$ITER/added-vocab $SAVE_CP/$ITER $ITER
    python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type llama2 --load_dir $SAVE_CP/$ITER --save_dir $SAVE_CP/$ITER/hf --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1
    python3 ../Megatron-LLM/weights_conversion/megatron_to_hf.py --input_dir $SAVE_CP/$ITER/hf --output_dir $SAVE_CP/$ITER/hf --model llama2 --vocab_file ./tokenizer.model --num_output_shards 3 --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
    rm -rf $SAVE_CP/$ITER/iter*
    rm -rf $SAVE_CP/$ITER/latest*
    rm -rf $SAVE_CP/$ITER/hf/iter*
    rm -rf $SAVE_CP/$ITER/added-vocab
fi

conda deactivate
CONDA_ENV_PATH='/home/cse/btech/cs1200389/.conda/envs/matllama-inference'
conda activate ${CONDA_ENV_PATH}
module unload compiler/cuda/12.3/compilervars
module load compiler/cuda/11.0/compilervars

sh ft_eval.sh $SAVE_CP/$ITER/hf