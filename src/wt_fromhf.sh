# sh wt_fromhf.sh <hf model path> <save path> <llama2/llama3> <7/8>
python3 ../Megatron-LLM/weights_conversion/hf_to_megatron.py $3 --size $4 --model-path $1 --out $2
python3 ../Megatron-LLM/tools/checkpoint_util.py --model_type $3 --load_dir $2 --save_dir $2 --target_tensor_parallel_size 1 --target_pipeline_parallel_size 4
rm -r $2/release/mp_rank_00