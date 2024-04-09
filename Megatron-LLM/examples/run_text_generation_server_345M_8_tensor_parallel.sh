#!/bin/bash
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=/scratch/cse/btech/cs1200448/meditron/7b
VOCAB_FILE=/scratch/cse/btech/cs1200448/meditron/7b/tokenizer.model
# MERGE_FILE=<Path to merges.txt (e.g. /gpt2-merges.txt)>

# pip install flask-restful

python -m torch.distributed.launch $DISTRIBUTED_ARGS ../tools/run_text_generation_server.py   \
       --tensor_model_parallel_size 8  \
       --pipeline_model_parallel_size 1  \
       --load ${CHECKPOINT}  \
       --fp16  \
       --micro_batch_size 1  \
       --seq_length 1024  \
       --out_seq_length 1024  \
       --temperature 1.0  \
       --vocab_file $VOCAB_FILE  \
       # --merge_file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42
