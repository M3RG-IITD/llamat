#! /bin/bash

SIZE=7
GPUS_PER_NODE=4
TP=1
PP=4
MICRO_BATCH=2 # per dp batch size.
GLOBAL_BATCH=1024 # micro * dp * grad_accum
RANK=0
N_NODES=1
ADDR=localhost
EXP_NAME=$1

MODEL='llama2'

LR="3e-4"
MODEL_CONFIG=${MODEL}-${SIZE}b-tp$TP-pp$PP

# LOAD_CHECKPOINT_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/llama3_trained_vaibhav
# LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/hf-to-meditron-weights/8b
# LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/llama3
LOAD_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/hf-to-meditron-weights/7b
# SAVE_CHECKPOINT_PATH=/scratch/cse/btech/cs1200389/llama3-debug/${EXP_NAME}
SAVE_CHECKPOINT_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/${EXP_NAME}
TENSORBOARD_PATH=/scratch/cse/btech/cs1200448/MatLlama/meditron-checkpoints/${EXP_NAME}/tensorboards/${MODEL_CONFIG}

# TRAIN_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/train/train_text_document
# VALID_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/val/val_text_document
# TEST_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/redp_val_text_document

TRAIN_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/mscd_text_document
VALID_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/redP_val_text_document
TEST_DATA_PATH=/scratch/civil/phd/cez198233/vaibhav_nlp/pretrain_datasets/bin/redP_val_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank $RANK --master_addr $ADDR"

TOKENIZER=SentencePieceTokenizer

EXTRA_ARGS='--vocab_file=/scratch/cse/btech/cs1200448/llama-weights/7b/tokenizer.model --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --clip_grad 1.0'

# EXTRA_ARGS='--vocab_file=/scratch/cse/btech/cs1200448/hf-to-meditron-weights/8b/tokenizer.model --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --clip_grad 1.0'

# EXTRA_ARGS='--vocab_file=/scratch/cse/btech/cs1200389/llama3-debug/tokenizer.model --use_rms_norm --glu_activation swiglu --no_tie_embed_logits --num_layers 32 --hidden_size 4096 --num_attention_heads 32 --ffn_hidden_size 14336 --max_position_embeddings 8192 --num_attention_heads_kv 8 --make_vocab_size_divisible_by 128256' # llama3 config
         
SEQ_LEN=2048 # ctx length
EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
if (( $SIZE > 13 )); then  # llama 2, 34B and 70B
        LR="1.5e-4"
fi

TRAIN_TOKENS=30135507860
EVAL_TOKENS=514977472
TEST_TOKENS=68328730

TRAIN_SEQS=$((TRAIN_TOKENS/SEQ_LEN))
EVAL_SEQS=$((EVAL_TOKENS/SEQ_LEN))

TRAIN_ITERS=$((TRAIN_SEQS/GLOBAL_BATCH))
EVAL_ITERS=$((EVAL_SEQS/GLOBAL_BATCH))
TEST_ITERS=$((TEST_TOKENS/GLOBAL_BATCH))
EVAL_ITERS=10

COMMON_ARGS="$COMMON_ARGS --use_flash_attn --no_bias_gelu_fusion
		--seq_length $SEQ_LEN
		--log_interval 1 --eval_interval 150 --save_interval 150
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

CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 torchrun $DISTRIBUTED_ARGS /home/cse/btech/cs1200389/MatLlama/MatLLaMA/Megatron-LLM/finetune.py \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP  \
	--load $LOAD_CHECKPOINT_PATH \
	--save $SAVE_CHECKPOINT_PATH \
	--tensorboard_dir $TENSORBOARD_PATH \
	--model_name $MODEL \
	--tokenizer_type $TOKENIZER \
	--bf16 \
	--global_batch_size $GLOBAL_BATCH \
	--micro_batch_size $MICRO_BATCH \
	$EXTRA_ARGS \
	$COMMON_ARGS #> logs/$1_${current_datetime}.txt