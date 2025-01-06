# Contains evaluation code for MatNLP and MatSIE datasets

## Instructions to run matNLP evaluations 

        bash ft_eval_downstream.sh <Checkpoint_path> <GPU_number> <output_name1> <output_name2>

the output and error file will be stored in the same directory and their exact names can be found from ft_eval_donwstream.sh file.

## Instructions to run structured information extraction evaluations:

### Generating the output pickle file:
        
        python3 {doping, mof1, mof2, discomat}_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>                               

Output will be stored as <SAVE_NAME_PREFIX>_{doping, mof1, mof2, discomat}_test.pkl in the same folder 

### running evaluation on the output file:
        
        python3 {doping, mof1, mof2, discomat}_eval.py <SAVE_NAME_PREFIX>                               

This will print the output to the screen along the metrics discussed in the paper.
