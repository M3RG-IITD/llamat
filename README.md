# LLaMat 
This repo contains all the data and code related to our paper [Foundational Large Language Models for Materials Research
](https://arxiv.org/abs/2412.09560). 

## Table of contents
---
- [Overview](#overview)
- [Evaluation](#evaluation)
- [plots](plots)
- [src](src)  


---
#Overview

- [evaluation_codes](evaluation_codes)
- [plots](plots)
- [src](src)
- 
---

## Evaluation
for running the benchmark evaluations on our datasets. to run, first open the [evaluation_codes](evaluation_codes) directory and follow the given instructions.
### Instructions to run matNLP evaluations 

        bash ft_eval_downstream.sh <Checkpoint_path> <GPU_number> <output_name1> <output_name2>

the output and error file will be stored in the same directory and their exact names can be found from ft_eval_donwstream.sh file.

## Instructions to run structured information extraction evaluations:

### Generating the output pickle file:
        
        python3 {doping, mof1, mof2, discomat}_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>                               

Output will be stored as <SAVE_NAME_PREFIX>_{doping, mof1, mof2, discomat}_test.pkl in the same folder 

### running evaluation on the output file:
        
        python3 {doping, mof1, mof2, discomat}_eval.py <SAVE_NAME_PREFIX>                               

This will print the output to the screen along the metrics discussed in the paper.


# Documentation
The codebase makes use of the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) library for efficient training of LLMs. Go through their documentation to understand the basics. The environment for using our codebase is same as the one for Megatron-LLM.
## File Structure
- [src](src)
  Contains code to pretrain and fine-tune LLMs that have the LLaMA-2 or LLaMA-3 architecture.
- [preprocess](preprocess)
  Contains code that was used to extract text from research papers for the corpus.
- [plots](plots)
  Code used for creating the plots used in the paper
- [evaluation_codes](evaluation_codes)
  Contains code for running benchmark evaluations

## Instruction Fine-Tuning
#### Command:
```
sh ft_pipeline.sh <load_model_path> <save_model_path> <model_iteration_to_finetune> <train_path>\
<val_path> <epochs> <number of docs in train set> <log_file_name> <llama2/llama3> <port number>
```
The files that are responsible for IFT:
- `ft_pipeline.sh`
- `finetune.sh`
- `ft_sft.py`
- `ft_sft.sh`

Arguments flow from top to bottom in the above list.

The weights of the input model must be stored in the Megatron format. To convert model weights from the HuggingFace format to Megatron format, `wt_fromhf.sh` is used. For the reverse conversion `wt_tohf.sh` is used. The model weights resulting from IFT are stored in the HF format to facilitate inference.

## Inference and Evaluation
The environment for inference requires the [VLLM](https://docs.vllm.ai/en/stable/getting_started/installation.html) library.
Evaluation and inference is done using `ft_eval.sh`.
