# Contains evaluation code for MatNLP and MatSIE datasets

### Environment setup downstream evaluation
We also provide a complete environment that can be used for inference on our downstream tasks, however note that it is different from the one used for training. 

```bash
conda env create -f infer_env_downstream.yaml
conda activate infer_env_downstream
```

## Instructions to run matNLP evaluations 
make sure you are in the evaluation_codes directory. then run the following

        bash ft_eval_downstream.sh <Checkpoint_path> <GPU_number> <output_name1> <output_name2>

<output_name1>_<output_name2> will be the suffix for the output and error files. 
the output and error files will be stored in the same directory and their exact names can be found from `evaluation_codes/ft_eval_donwstream.sh` file.

## Instructions to run structured information extraction evaluations:

### Generating the output pickle file:
        
        python3 {doping, mof1, mof2, discomat}_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>                               

Output will be stored as <SAVE_NAME_PREFIX>_{doping, mof1, mof2, discomat}_test.pkl in the same folder. 
here is an example command,

        python3 mof1_run.py 0 ../models/llamat3chat_hf llamat3chat
        
running the above code will run the model provided on the doping tasks and produce an output pickle file with the name llamat3chat_mof1_test.pkl, which can be passed to the evaluation function.

### running evaluation on the output file:
        python3 {doping, mof1, mof2, discomat}_eval.py <SAVE_NAME_PREFIX>                           
        
This will print the output to the screen along the metrics discussed in the paper.
here is an example command to evaluate mof1 tasks,

        python3 mof1_eval.py llamat3chat
        
this will search for llamat3chat_mof1_test.pkl file in the same directory, and give the results for the model on the mof1 (General materials science) tasks. 
      
---
