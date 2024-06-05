import json

file1="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/p1_10.jsonl"
file2="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/p2_10.jsonl"
file3="/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/p3_10.jsonl"
file4="/scratch/cse/btech/cs1200448/MatLlama/redP_fresh/train.json"
output = "/scratch/civil/phd/cez198233/vaibhav_nlp/corpus/corpus_training/trainFirst.jsonl"

chunk_size1=1000
chunk_size2=200
chunk_size3=100
chunk_size4=1300

f1 = open(file1, 'r').readlines()
f2 = open(file2, 'r').readlines()
f3 = open(file3, 'r').readlines()
f4 = open(file4, 'r').readlines()

with open(output, 'w') as f:
    p1 = 0
    p2 = 0
    p3 = 0 
    p4 = 0
    while True:
        for j in range(p1, min(p1+chunk_size1, len(f1))):
            f.write(f1[j])
        for j in range(p2, min(p2+chunk_size2, len(f2))):
            f.write(f2[j])
        for j in range(p3, min(p3+chunk_size3, len(f3))):
            f.write(f3[j])
        for j in range(p4, min(p4+chunk_size4, len(f4))):
            f.write(f4[j])
        p1 += chunk_size1
        p2 += chunk_size2
        p3 += chunk_size3
        p4 += chunk_size4
        
        print(p1, p2, p3, p4)
        if p1>=len(f1) and p2>=len(f2) and p3>=len(f3) and p4>=len(f4):
            break