
## First download the orca dataset(https://huggingface.co/datasets/Open-Orca/OpenOrca/blob/main/3_5M-GPT3_5-Augmented.parquet) and convert it to jsonl format and store in ../../datasets/orca/orca_train.jsonl
## then we run these to preprocess the data based on llama3 and llama2

mkdir -p ../datasets/orca

sh preprocess_ift.sh llama3 ../datasets/orca/orca_train.jsonl ../datasets/orca/orca_train_llama3 
sh preprocess_ift.sh llama2 ../datasets/orca/orca_train.jsonl ../datasets/orca/orca_train_llama2 

##

## fill this path with the folder that contains the folder containing meditron weights. ideally to be put in ../checkpoints
in_checkpoint_path=

## the name of the folder containing weights, (inside $in_checkpoint_path)
weight_folder_name="release"

samples=576000

## output folder, can edit based on need.
out_checkpoint_path="../checkpoints/train_llamat3_orca576k"

sh zft_pipeline.sh \
$in_checkpoint_path \
$out_checkpoint_path \
$weight_folder_name \
../datasets/orca/orca_train_llama3 \
../datasets/orca/orca_train_llama3 \
1 \
$samples \
llama3_log \
llama3 \
9000