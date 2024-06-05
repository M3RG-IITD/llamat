## Usage: "sh preprocess_ift.sh <model> <inputpath with extension> <outputpath without extension>"

if [ "$1" = "llama2" ]; then
    tpath="./tokenizer_l2.model"
    ttype="SentencePieceTokenizer"
elif [ "$1" = "llama3" ]; then
    tpath="./tokenizer_l3.model"
    ttype="Tiktoken"
fi

python ../Megatron-LLM/tools/preprocess_instruct_data.py \
	--input=$2 \
	--output_prefix=$3 \
	--tokenizer_type=$ttype \
	--vocab_file=$tpath \
	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	--chunk_size=32 \
	--workers=32 \
	--system_key=system\
	--question_key=question\
	--answer_key=answer\
