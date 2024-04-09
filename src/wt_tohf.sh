python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type llama2 --load_dir $1 --save_dir $2 --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1 --load_iters $3
python3 ../Megatron-LLM/weights_conversion/megatron_to_hf.py --input_dir $2 --output_dir $2 --model llama2 --vocab_file ./tokenizer.model --num_output_shards 3 --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
rm -r iter*