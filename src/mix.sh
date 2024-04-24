#!/bin/bash

file1="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/train_10.jsonl"
file2="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/redP.jsonl"

# Define the number of lines to extract in each chunk
chunk_size=10000

# Initialize pointers for both files
pointer_file1=1
pointer_file2=1

# Loop until end of files is reached
while [[ $pointer_file1 -ne 0 || $pointer_file2 -ne 0 ]]; do
    # Extract chunk from file 1
    chunk_file1=$(tail -n +$pointer_file1 "$file1" | head -n $chunk_size)
    pointer_file1=$((pointer_file1 + chunk_size))

    # Extract chunk from file 2
    chunk_file2=$(tail -n +$pointer_file2 "$file2" | head -n $chunk_size)
    pointer_file2=$((pointer_file2 + chunk_size))

    # Check if either file has reached the end
    if [[ -z $chunk_file1 && -z $chunk_file2 ]]; then
        break
    fi

    # Concatenate chunks and write to concatenated.json
    echo "$chunk_file1" >> /scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/trainFirst.jsonl
    echo "$chunk_file2" >> /scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/trainFirst.jsonl
done
