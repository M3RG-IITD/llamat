#!/bin/bash

## This script sets up the dataset to be used during training and testing. 

sh preprocess_ift.sh llama3 ../datasets/final_paper_ift/train_extra_SIE.jsonl ../datasets/final_paper_ift/train_extra_SIE_llama3
sh preprocess_ift.sh llama2 ../datasets/final_paper_ift/train_extra_SIE.jsonl ../datasets/final_paper_ift/train_extra_SIE_llama2
sh preprocess_ift.sh llama3 ../datasets/final_paper_ift/val_extra_SIE.jsonl ../datasets/final_paper_ift/val_extra_SIE_llama3
sh preprocess_ift.sh llama2 ../datasets/final_paper_ift/val_extra_SIE.jsonl ../datasets/final_paper_ift/val_extra_SIE_llama2
sh preprocess_ift.sh llama3 ../datasets/final_paper_ift/test_extra_SIE.jsonl ../datasets/final_paper_ift/test_extra_SIE_llama3
sh preprocess_ift.sh llama2 ../datasets/final_paper_ift/test_extra_SIE.jsonl ../datasets/final_paper_ift/test_extra_SIE_llama2
