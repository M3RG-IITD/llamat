
# Requires the vllm library

# $1 checkpoint path as $1, llamat-2-chat or llamat3-chat
ckp_path=$1
# $2 = GPU number , $3 and $4 for naming 
out_file="out_$2_$3_$4"
err_file="err_$2_$3_$4"

CUDA_VISIBLE_DEVICES=$2 
#python3 ft_eval.py --checkpoint $ckp_path --valfile ../datasets/downstream/val/val.jsonl --mem_util 0.9 --num_gpu 1 --num_seeds 3 >> $out_file 2>> $err_file
python3 ft_eval.py --checkpoint $ckp_path --valfile test_downstream.jsonl --mem_util 0.9 --num_gpu 1 --num_seeds 1 >> $out_file 2>> $err_file
