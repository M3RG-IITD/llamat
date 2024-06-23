# Documentation
The codebase makes use of the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) library for efficient training of LLMs. Go through their documentation to understand the basics. The environment for using our codebase is same as the one for Megatron-LLM.
## File Structure
- src
  Contains code to pretrain and fine-tune LLMs that have the LLaMA-2 or LLaMA-3 architecture.
- preprocess
  Contains code that was used to extract text from research papers for the corpus.
## Pretraining 

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
