#! /bin/bash

# sh pretrain.sh <expname> <llama2/llama3> <size>

GPUS_PER_NODE=8
TP=1
PP=4
MICRO_BATCH=2 # per dp batch size.
GLOBAL_BATCH=1024 # micro * dp * grad_accum
RANK=0
N_NODES=1
ADDR=localhost
EXP_NAME=$1
MODEL=$2
SIZE=$3

LR="3e-4"

# LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/hf-to-meditron-weights/7b
# LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/hf-to-meditron-weights/8b-testing
LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/llama3_pmc_ds
# LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/llama2_debug
SAVE_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/${EXP_NAME}

TRAIN_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/train_pmc_llama3_text_document
VALID_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/val_llama3_text_document
TEST_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/redp_val_llama3_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank $RANK --master_addr $ADDR"

if [ "$MODEL" = "llama2" ]; then
    tpath="./tokenizer_l2.model"
    ttype="SentencePieceTokenizer"
elif [ "$MODEL" = "llama3" ]; then
    tpath="./tokenizer_l3.model"
    ttype="Tiktoken"
fi

EXTRA_ARGS="--vocab_file=$tpath --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --clip_grad 1.0"

# EXTRA_ARGS="--vocab_file=$tpath --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --num_layers 32 --hidden_size 4096 --num_attention_heads 32 --ffn_hidden_size 14336 --max_position_embeddings 8192 --num_attention_heads_kv 8 --make_vocab_size_divisible_by 128256" # llama3 config
         
SEQ_LEN=2048 # ctx length
EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
if (( $SIZE > 13 )); then  # llama 2, 34B and 70B
        LR="1.5e-4"
fi

# these values will also change for llama2 and llama3. maybe for the dataset too?

if [ "$MODEL" = "llama2" ]; then
    TRAIN_TOKENS=31537662582 # redP + papers + mscd + cif
    EVAL_TOKENS=531091878 # cif + paper validation
    TEST_TOKENS=414815173 # redP validation
elif [ "$MODEL" = "llama3" ]; then
    TRAIN_TOKENS=27146746178 # redP + papers + mscd + cif
    EVAL_TOKENS=455454671 # cif + paper validation
    TEST_TOKENS=347375685 # redP validation
fi

TRAIN_SEQS=$((TRAIN_TOKENS/SEQ_LEN))
EVAL_SEQS=$((EVAL_TOKENS/SEQ_LEN))
TEST_SEQS=$((TEST_TOKENS/SEQ_LEN))

TRAIN_ITERS=$((TRAIN_SEQS/GLOBAL_BATCH))
EVAL_ITERS=$((EVAL_SEQS/GLOBAL_BATCH))
TEST_ITERS=$((TEST_SEQS/GLOBAL_BATCH))

EVAL_ITERS=20

COMMON_ARGS="$COMMON_ARGS --use_flash_attn --no_bias_gelu_fusion
		--seq_length $SEQ_LEN
		--log_interval 1 --eval_interval 150 --save_interval 300
		--use_checkpoint_args --hidden_dropout 0.0
		--position_embedding_type rotary
		--no_bias_dropout_fusion --attention_dropout 0.0
		--adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5
		--lr_decay_style cosine --lr_warmup_iters 2000 --lr $LR --min_lr 3e-5
		--weight_decay 0.1 --sequence_parallel --recompute_granularity selective
		--log_validation_ppl_to_tensorboard
        --log_memory_to_tensorboard
        --log_timers_to_tensorboard
		--num_workers 0 --dataloader_type cyclic
		--train_data_path $TRAIN_DATA_PATH
		--valid_data_path $VALID_DATA_PATH
        --test_data_path $TEST_DATA_PATH
		--eval_iters $EVAL_ITERS
		--train_iters $TRAIN_ITERS"

echo 
echo Settings:
echo RANK=$RANK
echo ADDR=$ADDR
echo N_NODES=$N_NODES
echo DATA_PATH=$TRAIN_DATA_PATH
echo CHECKPOINT_PATH=$LOAD_CHECKPOINT_PATH
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo MICRO_BATCH=$MICRO_BATCH
echo GLOBAL_BATCH=$GLOBAL_BATCH
echo EVAL_ITERS=$EVAL_ITERS
echo

current_datetime=$(date +"%y_%m_%d_%H_%M")

CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 torchrun $DISTRIBUTED_ARGS ../Megatron-LLM/finetune.py \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP  \
	--load $LOAD_CHECKPOINT_PATH \
	--save $SAVE_CHECKPOINT_PATH \
	--model_name $MODEL \
	--tokenizer_type $ttype \
	--bf16 \
	--global_batch_size $GLOBAL_BATCH \
	--micro_batch_size $MICRO_BATCH \
	$EXTRA_ARGS \
	$COMMON_ARGS > logs/$1_${current_datetime}.txt