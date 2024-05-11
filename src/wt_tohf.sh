# sh wt_tohf.sh <meditron model path> <model save path> <llama2/llama3> <iter>

if [ "$3" = "llama2" ]; then
    tokenizer="./tokenizer_l2.model"
elif [ "$3" = "llama3" ]; then
    tokenizer="./tokenizer_l3.model"
fi

python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type $3 --load_dir $1 --save_dir $2 --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1 --load_iters $4

# gotta do the tokenizer thingy for when there are extra vocabs, ie in finetuning.
python3 ../Megatron-LLM/weights_conversion/megatron_to_hf.py --input_dir $2 --output_dir $2 --model $3 --vocab_file $tokenizer --num_output_shards 3 #--vocab_extra_ids_list "<|im_start|>,<|im_end|>"