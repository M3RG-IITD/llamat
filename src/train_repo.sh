echo "Training all finally"

echo "Setting up dataset"
bash dataset_setup.sh

## makes a directory for storing logs.
mkdir -p logs 

## replace this with where you stored the ift dataset. 
ift_dataset_path="/home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/ift/final_paper_ift"

## replace this with the location of all the checkpoints stored. 
checkpoints_folder="/scratch/civil/faculty/krishnan/cs1200448.scratch.hf-to-meditron-weights"

## the name of the checkpoint on which ift training is conducted. Should be the one you get after training on openorca
original_checkpoint="cerebras_13812_llama38b"


## instead of 'release', you can put the name of the folder which contains the meditron weights in the checkpoint folder. Usually it is the iteration number in which case just write it as a plain number without leading zeroes. 
meditron_weights_folder="release"




function train_additional()
{
    SECONDS=0;
    echo "started training on $1 for datapath = $train_file" >> logs/train_results.txt


    path=$1
    llama=$2
    numiter=$3
    finalds=$4"_train_"$5"epochs_"$6  #the name of the final 
    epochs=$5
    train_file=train_"$6"_$llama
    val_file=val_"$6"_$llama
    datasize=$7
    echo "further additional ift training on $path" #>> logs/train_all.log
    echo "final save location : $finalds"

    #### Final downstream training.
    sh ft_pipeline.sh \
    $path \
    $checkpoints_folder/"$finalds" \
    $numiter \
    $ift_dataset_path/$train_file \
    $ift_dataset_path/$val_file \
    $epochs \
    $datasize \
    log_"$finalds" \
    $llama \
    9000

    ## for multiple training: 
    echo "completed training on $1 for datapath = $6 in time: $SECONDS" >> logs/final_train_results.txt
}

function train_on_dataset()
{
epochs=$1
datapath=$2
datasize=$3
checkpoints=/scratch/civil/faculty/krishnan/cs1200448.scratch.hf-to-meditron-weights

train_additional "$checkpoints/$original_checkpoint" "llama3" "$meditron_weights_folder" "llamat3_ift" $epochs $datapath $datasize

}

epochs=2
dataset="extra_SIE"

train_on_dataset $epochs $dataset 18563 #the final training call to train the model on our downstream and information extraction datasets. 

echo "FINISHED WITH TRAINING ALL VARIATIONS"
echo "*******************************************************************************"

#first we need to copy tokenizers to the llama3 models because its not properly copied for them. Then after that we need to further continue by running these on the individual datasets that we have. and we also need to get the outputs from this and store that in a particular way
