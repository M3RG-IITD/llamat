# LLaMat 
This repo contains all the data and code related to our paper [Foundational Large Language Models for Materials Research
](https://arxiv.org/abs/2412.09560). 

## Table of contents
---
- [Overview](#overview)
- [Pretraining](#pretraining)
- [File Structure](#file-structure)
- [Inference and Evaluation](#inference-and-evaluation)
- [Instruction finetuning](instruction-finetuning)

---
## Overview
We performed domain adaptation of the models LLaMA-3 and LLaMA-2 for use in material science, via continued pretraining followed by instruction finetuning on material science and chemistry datasets. 
<
<img align="left" src="https://github.com/user-attachments/assets/461a6aba-6321-45c8-a893-eb1e2b4f4db2" alt="overview" style="width:50%; height:auto;">

LLaMat overview and usage tasks.

for full results please look at our paper [Foundational Large Language Models for Materials Research
](https://arxiv.org/abs/2412.09560). The models can be downloaded from [https://huggingface.co/m3rg-iitd](https://huggingface.co/m3rg-iitd). The codebase makes use of the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) library for efficient training of LLMs. Go through their documentation to understand the basics. The environment for using our codebase is same as the one for Megatron-LLM.

---
## Pretraining
Pretraining was performed on a text corpus of total 30B tokens, interleaved in the following way:

1. 10M research paper tokens taken from Elsevier and Springer publications followed by 0.1M Red Pajama tokens 
2. 30M Matsci community discourse tokens included in the last 3B (10%) of the dataset in
100:1 ratio.
   
The pretraining was performed on a Cerebras-CS2 cluster and supported by Edinburgh International Data Facility (EIDF) and University of Edinburgh. 

---
## File Structure
- [src](src)
  Contains code to pretrain and fine-tune LLMs that have the LLaMA-2 or LLaMA-3 architecture.
- [preprocess](preprocess)
  Contains code that was used to extract text from research papers for the corpus, from elsevier and springer.
- [plots](plots)
  Code used for creating the plots used in the paper
- [evaluation_codes](evaluation_codes)
  Contains code for running benchmark evaluations
  
---

## Inference and Evaluation
for running the benchmark evaluations on our datasets. to run, first open the [evaluation_codes](evaluation_codes) directory and follow the given instructions. The environment for inference for matNLP tasks requires the [VLLM](https://docs.vllm.ai/en/stable/getting_started/installation.html) library.

### Instructions to run matNLP evaluations 

        bash ft_eval_downstream.sh <Checkpoint_path> <GPU_number> <output_name1> <output_name2>

the output and error file will be stored in the same directory and their exact names can be found from ft_eval_donwstream.sh file.

### Instructions to run structured information extraction evaluations:

### Generating the output pickle file:
        
        python3 {doping, mof1, mof2, discomat}_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>                               

Output will be stored as <SAVE_NAME_PREFIX>_{doping, mof1, mof2, discomat}_test.pkl in the same folder 

### running evaluation on the output file:
        
        python3 {doping, mof1, mof2, discomat}_eval.py <SAVE_NAME_PREFIX>                               

This will print the output to the screen along the metrics discussed in the paper.



## Instruction finetuning
### Command:
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
The Instruction finetuning process was performed on 8 Nvidia-A100 80GB GPUs via IIT Delhi's High Performance Computing facility. 

The weights of the input model must be stored in the Megatron format. To convert model weights from the HuggingFace format to Megatron format, `wt_fromhf.sh` is used. For the reverse conversion `wt_tohf.sh` is used. The model weights resulting from IFT are stored in the HF format to facilitate inference.
