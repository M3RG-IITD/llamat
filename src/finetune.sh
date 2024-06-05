port=${11}
ttype=${10}
python3 ft_sft.py --checkpoint $1 --save_checkpoint_dir $2 --epochs $6 --save_interval 500 --load_iters $3 --traindata $4 --valdata $5 --numdocs $7 --runname $8 --port $port --tpath $9 --ttype $ttype --size ${12}