## Usage: "sh pre_process.sh train/val" to convert train/val_corpus.json to .bin and .idx files, both of which are needed as input to megatronLLM training

python ../Megatron-LLM/tools/preprocess_data.py --input=./datasets/${1}.jsonl \
	--output_prefix=./datasets/$1 \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=./tokenizer.model \
	--chunk_size=32 \
	--workers=16 \
	--no_new_tokens
