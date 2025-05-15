#!/bin/bash


## replace ../../Megatron with the actual path. 

# return 19942 #hb downstream train
# return 170594 #hb downstream val
# return 43841 #hb ift train
# return 2307 #hb ift val
# return 10581 #downstream older
# return 52597 # ift train
# return 48098 # ift of hb and msqa

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_adding_math/7000 /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_30B_448k_7500x3_hb_msqa 360 ../datasets/ift/bin/train_refined ../datasets/ift/bin/val_refined 3 48098 llama2_30B_448k_7500x3_hb_msqa llama2 29500 > ./logs/llama2_30B_448k_7500x3_hb_msqa.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_30B_0_xk_oldift/14369 /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_30B_0_64k_ds_oldift 1000 ../datasets/downstream/train ../datasets/downstream/val 1 10581 llama2_30B_0_64k_ds_oldift llama2 29500 > ./logs/llama2_30B_0_64k_ds_oldift.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/llama2-final-pretraining /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_30B_xk 14369 /home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/ift/bin/stage1_train /home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/ift/bin/stage1_val 1 800000 llama2_30B_xk llama2 > ./logs/llama2_30B_xk.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/santiago /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/santiago-ift 1500 ../datasets/ift/compare_hb/train ../datasets/ift/compare_hb/val 3 10581 run_name > ./logs/run_name.txt

# for honeybee
# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/hf-to-meditron-weights/7b /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/hb-llama release ../datasets/downstream/train_hb ../datasets/downstream/val_hb 3 

# sh ft_pipeline.sh /home/civil/faculty/krishnan/NLP/matllama /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/santiago-10B 5000 ../datasets/downstream/train ../datasets/downstream/val 1 10581

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/cif /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llamatcif_iftcif 751 /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/train /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val 1 5896850 santiago_ift_cif llama2 29500 > log_ift_cif2.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/hf-to-meditron-weights/7b /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama2_iftcif 1650 /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/train /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val 1 5896850 santiago_ift_cif llama2 29500 > log_ift_cif2.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/cif_inter_2 /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llamat_cifpt_iftcif 2559 /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/train /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val 1 6941865 santiago_ift_cif llama2 29500 > log_ift_cif_new.txt

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/cif_inter_2 /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llamat_cifpt_iftcif_new 2559 /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/train /scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val 1 3453189 santiago_ift_cif llama2 29500

# sh ft_pipeline.sh /scratch/cse/btech/cs1200448/hf-to-meditron-weights/8b /scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llama3_downstream 1 /home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/downstream/bin/train_llama3 /home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/downstream/bin/val_llama3 1 12300 llama3_downstream llama3 29500 > log_ift_cif_new.txt

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
            elif [[ -z $MODEL ]]; then
                MODEL="$1"
            elif [[ -z $PORT ]]; then
                PORT="$1"
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
echo $MODEL

source ~/.bashrc
CONDA_ENV_PATH='/home/cse/btech/cs1200389/.conda/envs/matllama2.0'
conda activate ${CONDA_ENV_PATH}

module load compiler/gcc/11.2.0
module unload compiler/cuda/11.0/compilervars
module load compiler/cuda/12.3/compilervars

if [ "$MODEL" = "llama2" ]; then
    tpath="./tokenizer_l2.model"
    ttype="SentencePieceTokenizer"
    vocabsize=32007
    size=7
elif [ "$MODEL" = "llama3" ]; then
    tpath="./tokenizer_l3.model"
    ttype="Tiktoken"
    vocabsize=128258
    size=8
fi

if $TRAIN; then
    echo "========Adding embeddings in the model for <|im_start|> and <|im_end|>========"
    if [ "$ITER" = "release" ]; then
        python3 ../../Megatron-LLM/tools/checkpoint_util.py --model_type $MODEL --load_dir $LOAD_CP --save_dir $SAVE_CP/$ITER/added-vocab --true_vocab_size $vocabsize
    else
        python3 ../../Megatron-LLM/tools/checkpoint_util.py --model_type $MODEL --load_dir $LOAD_CP --save_dir $SAVE_CP/$ITER/added-vocab --true_vocab_size $vocabsize --load_iters $ITER
    fi
    
    echo "========Finetuning the model========"
	sh ./finetune.sh $SAVE_CP/$ITER/added-vocab $SAVE_CP/$ITER $ITER $TRAINDATA $VALDATA $EPOCH $NUMDOCS $FT_SAVENAME $tpath $ttype $PORT $size
    
    echo "========Converting the model to tp1pp1========"
    python3 ../../Megatron-LLM/tools/checkpoint_util.py --model_type $MODEL --load_dir $SAVE_CP/$ITER --save_dir $SAVE_CP/$ITER/hf --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1
    
    echo "========Converting the model to hf format========"
    python3 ../../Megatron-LLM/weights_conversion/megatron_to_hf.py --input_dir $SAVE_CP/$ITER/hf --output_dir $SAVE_CP/$ITER/hf --model $MODEL --vocab_file /scratch/cse/btech/cs1200448/llama-weights/7b/tokenizer.model --num_output_shards 3 --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
    # rm -rf $SAVE_CP/$ITER/iter*
    # rm -rf $SAVE_CP/$ITER/latest*
    
    echo "========Deleting tp1pp1========"
    rm -rf $SAVE_CP/$ITER/hf/iter*
    
    echo "========Deleting the expanded vocab base model========"
    rm -rf $SAVE_CP/$ITER/added-vocab
fi

conda deactivate
CONDA_ENV_PATH='/home/cse/btech/cs1200389/.conda/envs/matllama-inference'
conda activate ${CONDA_ENV_PATH}
module unload compiler/cuda/12.3/compilervars
module load compiler/cuda/11.0/compilervars

# sh ft_eval.sh $SAVE_CP/$ITER/hf $FT_SAVENAME
