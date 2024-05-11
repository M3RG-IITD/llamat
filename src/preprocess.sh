## Usage: "sh preprocess.sh <model> <inputpath with extension> <outputpath without extension>"

if [ "$1" = "llama2" ]; then
    tpath="./tokenizer_l2.model"
    ttype="SentencePieceTokenizer"
elif [ "$1" = "llama3" ]; then
    tpath="./tokenizer_l3.model"
    ttype="Tiktoken"
fi

python ../Megatron-LLM/tools/preprocess_data.py --input=$2 \
	--output_prefix=$3 \
	--tokenizer_type=$ttype \
	--vocab_file=$tpath \
	--chunk_size=32 \
	--workers=16 \
	--no_new_tokens
