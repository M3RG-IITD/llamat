#!/bin/bash

jsonl_file="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/train.jsonl"

total_lines=$(wc -l < "$jsonl_file")
sample_size=$((total_lines*${1} / 100))

shuf -n "$sample_size" "$jsonl_file" > /scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/train${1}.jsonl