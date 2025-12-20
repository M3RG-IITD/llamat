# LLaMat 
This repo contains all the data and code related to our paper [Foundational Large Language Models for Materials Research
](https://arxiv.org/abs/2412.09560). 

## Table of contents
---
- [Overview](#overview)
- [File Structure](#file-structure)
- [Pretraining](#pretraining)
- [Inference and Evaluation](#inference-and-evaluation)
- [Instruction finetuning](#instruction-finetuning)

---
## Overview
We performed domain adaptation of the models LLaMA-3 and LLaMA-2 for use in material science, via continued pretraining followed by instruction finetuning on material science and chemistry datasets. 

<table>
  <tr>
    <td width="45%">
      <figure>
        <img src="https://github.com/user-attachments/assets/461a6aba-6321-45c8-a893-eb1e2b4f4db2" alt="overview" width="100%"/>
        <figcaption align="center"><b>LLaMat overview </b></figcaption>
      </figure>
    </td>
    <td width="55%">
      <figure>
        <img src="https://github.com/user-attachments/assets/e6083da0-f751-4b05-ad00-299257f935fa" width="100%"/>
        <figcaption align="center"><b>Results on MatNLP dataset</b></figcaption>
      </figure>
      <br>
      <figure>
        <img src="https://github.com/user-attachments/assets/28e44058-792f-4aa1-b403-d77588d2c48f" alt="results" width="100%"/>
        <figcaption align="center"><b>Results on structured information extraction tasks</b></figcaption>
      </figure>
    </td>
  </tr>
</table>

for detailed results please look at our paper [Foundational Large Language Models for Materials Research
](https://arxiv.org/abs/2412.09560). The models can be downloaded from [https://huggingface.co/m3rg-iitd](https://huggingface.co/m3rg-iitd). The codebase makes use of the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) library for efficient training of LLMs. Go through their documentation to understand the basics. The environment for using our codebase is same as the one for Megatron-LLM.


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

## Additional Documentation
- [agent/README.md](agent/README.md) - Interactive chat and NER agents for materials science applications
- [evaluation_codes/README.md](evaluation_codes/README.md) - Instructions for running MatNLP and MatSIE evaluations
- [visualizations/README.md](visualizations/README.md) - Streamlit dashboard for downstream evaluation analysis
- [src/cifs/crystal-text-llm/README.md](src/cifs/crystal-text-llm/README.md) - CIF generation dashboard and batch processing tools
- [src/cifs/crystal-text-llm/conditional_gen_eval/README.md](src/cifs/crystal-text-llm/conditional_gen_eval/README.md) - Conditional generation evaluation results and analysis

---
## Pretraining
Pretraining was performed on a text corpus of total 30B tokens, interleaved in the following way:

1. 10M research paper tokens taken from Elsevier and Springer publications followed by 0.1M Red Pajama tokens 
2. 30M Matsci community discourse tokens included in the last 3B (10%) of the dataset in
100:1 ratio.
the list of journals and DOIs of the research papers used can be accessed from [zenodo](https://zenodo.org/records/15101805?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2NDdiMDQ0LTQyOWMtNGJiZS1iZGVhLTY3OGI0MWZiYTQyOCIsImRhdGEiOnt9LCJyYW5kb20iOiI0Y2NjODVhMWJiMWM0YWQyMmZkZGZmNGRiYjA3NDkyZiJ9.a4apRHBEQzRs7gsFzbzM06spDgt1YCc-OMwNTNUMpc_6z5MXVTIaiAGpS4dQhd4Ib56p8RTKqOuIqXSBbr9bwQ)

### For setting up environment to install Megatron, please follow the instructions provided [here](https://github.com/M3RG-IITD/llamat/blob/main/installation_instructions.md)
---

## Inference and Evaluation
for running the benchmark evaluations on our datasets. to run, first open the [evaluation_codes](evaluation_codes) directory and follow the given instructions. The environment for inference for matNLP tasks requires the [VLLM](https://docs.vllm.ai/en/stable/getting_started/installation.html) library.

### Instructions to run matNLP evaluations 

        bash ft_eval_downstream.sh <Checkpoint_path> <GPU_number> <output_name1> <output_name2>

the output and error file will be stored in the same directory and their exact names can be found from ft_eval_donwstream.sh file.

### Instructions to run structured information extraction evaluations:

### Generating the output pickle file:
        
        python3 {doping, mof1, mof2, discomat}_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>                               

Output will be stored as <SAVE_NAME_PREFIX>_{doping, mof1, mof2, discomat}_test.pkl in the same folder. 
here is an example command,
        python3 mof1_run.py 0 ../models/llamat3chat_hf llamat3chat
running the above code will run the model provided on the doping tasks and produce an output pickle file with the name llamat3chat_mof1_test.pkl, which can be passed to the evaluation function.

### running evaluation on the output file:
        python3 {doping, mof1, mof2, discomat}_eval.py <SAVE_NAME_PREFIX>                           
        
This will print the output to the screen along the metrics discussed in the paper.
here is an example command,
        python3 mof1_eval.py llamat3chat
this will search for llamat3chat_mof1_test.pkl file in the same directory, and give the results for the model on the mof1 (General materials science) tasks. 
      
---
## Instruction finetuning

The weights of the input model must be stored in the Megatron format. To convert model weights from the HuggingFace format to Megatron format, `wt_fromhf.sh` is used. For the reverse conversion `wt_tohf.sh` is used. The model weights resulting from IFT are stored in the HF format to facilitate inference. After downloading the model from huggingface, this conversion is necessary for training.

### Step 1. OpenOrca finetuning.
We follow 2 step finetuning. First the model is trained on OpenOrca which is a generic IFT dataset. To run this finetuning, simply make the required path changes in `src/run_orca_ift.sh`, providing paths for input base model and output location, and the place where OpenOrca data is loaded. Further precise instructions are provided in `src/run_orca_ift.sh` itself. to run this, simply call bash on it while in the src directory. 

        bash run_orca_ift.sh

in the output path, the trained model will be present in huggingface format in the "release/hf" directory, as well as in meditron format in the "release/iter_0009000" directory.

### Step 2. Materials science specific finetuning. 
To run the next finetuning step, make the necessary path changes in the `src/train_repo.sh` file similarly, and make the base model as the output path given in the previous step. 
to start training, go in the src directory and run the following command
        bash train_repo.sh

this will also create the final model in both huggingface and meditron formats.

### General Command:
the following is the general command we use for finetuning within `src/run_orca_ift.sh` and `src/train_repo.sh`. 

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

---
## Acknowledgements
We used the codebase of [Meditron-LLM](https://github.com/epfLLM/meditron) for training our models on Nvidia A100 GPUs. 
We thank the High-Performance Computing (HPC) facility at IIT Delhi
for computational and storage resources. This work was partially supported by the Edinburgh International
Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh. The EIDF
provided access to Cerebras CS2 clusters which were used for performing pretraining on our models.
