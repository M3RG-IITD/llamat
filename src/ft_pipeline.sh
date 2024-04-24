#!/bin/bash

# return 19942 #hb downstream train
# return 170594 #hb downstream val
# return 43841 #hb ift train
# return 2307 #hb ift val
# return 10581 #downstream older
    
# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/santiago /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/santiago-ift 1500 ../datasets/ift/compare_hb/train ../datasets/ift/compare_hb/val 3

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/hf-to-meditron-weights/7b /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/hb-llama release ../datasets/downstream/train_hb ../datasets/downstream/val_hb 3 

TRAIN=true

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
            elif [[ -z $ITER ]]; then
                ITER="$1"
            elif [[ -z $TRAINDATA ]]; then
                TRAINDATA="$1"
            elif [[ -z $VALDATA ]]; then
                VALDATA="$1"
            elif [[ -z $EPOCH ]]; then
                EPOCH="$1"
            elif [[ -z $NUMDOCS ]]; then
                NUMDOCS="$1"
            elif [[ -z $FT_SAVENAME ]]; then
                FT_SAVENAME="$1"
            fi 
            ;;
    esac
    shift
done

echo $ITER
echo $LOAD_CP
echo $SAVE_CP
echo $TRAINDATA
echo $VALDATA
echo $EPOCH
echo $NUMDOCS
echo $FT_SAVENAME

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
	sh ./finetune.sh $SAVE_CP/$ITER/added-vocab $SAVE_CP/$ITER $ITER $TRAINDATA $VALDATA $EPOCH $NUMDOCS
    python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type llama2 --load_dir $SAVE_CP/$ITER --save_dir $SAVE_CP/$ITER/hf --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1
    python3 ../Megatron-LLM/weights_conversion/megatron_to_hf.py --input_dir $SAVE_CP/$ITER/hf --output_dir $SAVE_CP/$ITER/hf --model llama2 --vocab_file ./tokenizer.model --num_output_shards 3 --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
    # rm -rf $SAVE_CP/$ITER/iter*
    # rm -rf $SAVE_CP/$ITER/latest*
    rm -rf $SAVE_CP/$ITER/hf/iter*
    rm -rf $SAVE_CP/$ITER/added-vocab
fi

conda deactivate
CONDA_ENV_PATH='/home/cse/btech/cs1200389/.conda/envs/matllama-inference'
conda activate ${CONDA_ENV_PATH}
module unload compiler/cuda/12.3/compilervars
module load compiler/cuda/11.0/compilervars

sh ft_eval_hb.sh $SAVE_CP/$ITER/hf $FT_SAVENAME