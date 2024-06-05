# "/scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/santiago-10B-ift-stage2-downstream/1650/hf"
ckp_path=$1
python3 ft_eval.py --checkpoint $ckp_path --valfile ../datasets/downstream/val.jsonl --mem_util 0.4 --num_gpu 1 --num_seeds 3
