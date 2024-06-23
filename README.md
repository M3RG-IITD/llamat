# Documentation
The codebase makes use of the [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) library for efficient training of LLMs.
## File Structure
- src
  Contains code to pretrain and fine-tune LLMs that have the LLaMA-2 or LLaMA-3 architecture.
- preprocess
  Contains code that was used to extract text from research papers for the corpus.
## Pretraining 

## Instruction Fine-Tuning
Command:
```
sh ft_pipeline.sh <load_model_path> <save_model_path> <model_iteration_to_finetune> <train_path>\
<val_path> <epochs> <number of docs in train set> <log_file_name> <llama2/llama3> <port number>
```
