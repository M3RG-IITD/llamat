import json
import numpy as np
import pandas as pd
import string
import re
import collections
from sklearn.metrics import f1_score
import Levenshtein
import os
from tqdm import tqdm


def most_similar_answer(a,answer_set):
    a = a.strip().replace(' ', '')
    if(a in answer_set):
        return a
    dis = [Levenshtein.distance(a,x) for x in answer_set]
    idx = np.argmin(dis)
    return answer_set[idx]

def eval_atom_cnt(task, pred):
    # task = atom count
    return pred==task['output']

def eval_dimensions(task, pred):
    # task = dimensions, pred is a list
    if "lengths of the lattice vectors" in task['input']:
        lengths = [float(x) for x in task['output'].split(',')]
        mse = np.mean(abs(np.array(lengths) - np.array(pred))/np.array(lengths))
        return mse
    else:
        angles = [float(x) for x in task['output'].split(',')]
        mae = np.mean(abs(np.array(angles) - np.array(pred))/np.array(angles))
        return mae

def eval_atom_name(task, pred):
    # task = atom name
    return pred==task['output']

def eval_spacegroup(task, pred):
    # task = space group
    return pred==task['output']

def eval_cell_volume(task, pred):
    # task = cell_volume
    return abs(float(task['output'])-pred)/float(task['output'])

def eval_formula(task, pred):
    # task = formula
    return task['output']==pred

def eval_replace(task, pred):
    # task = replace
    answer = most_similar_answer(pred, ["Yes", "No"])
    return task['output']==answer

def eval_dimensions_sem(task, pred):
    lengths = [float(x) for x in task['output'].split('\n')[0].split()]
    mse = np.mean(abs(np.array(lengths) - np.array(pred[0]))/np.array(lengths))
    angles = [float(x) for x in task['output'].split('\n')[1].split()]
    mae = np.mean(abs(np.array(angles) - np.array(pred[1]))/np.array(angles))

    return mse, mae

def eval_infill_task(task, pred):
    return pred==task['output']

def eval_gen_format(pred):
    def trim_list(l):
        return [x.strip().replace('\n', '') for x in l if x]
    l = trim_list(pred.split('\n'))
    l1 = trim_list(l[0].split())
    l2 = trim_list(l[1].split())
    if len(l1)!=3 or len(l2)!=3:
        return 0
    matrix = l[2:]
    if len(matrix)%2:
        return 0
    for i in range(0, len(matrix), 2):
        l1 = trim_list(matrix[i].split())
        l2 = trim_list(matrix[i+1].split())
        if len(l1)!=1 or len(l2)!=3:
            return 0
        for x in l2:
            try:
                y = float(x)
            except:
                return 0
    return 1

valfile = []
with open("/scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

out_dict = dict()
tasks = ["atom count", "dimensions_synt", "atom name", "replace", "space group", "cell_volume", "formula", "dimensions_sem", "vol_calc"]
# "infill", "formula_compute", "conditional_generation", "element_generation"
for task in tasks:
    out_dict[task] = []
    
for _, sample in enumerate(valfile):
    task = sample['task']
    system = sample['system']

    with open(f'/home/cse/btech/cs1200448/MatLlama/cif_infer_outputs/{_}.txt', 'r') as f:
        output = f.read()

    if task in tasks:
        out_dict[task].append([output, sample])
    elif task=="dimensions":
        if "predict" not in system and "forecast" not in system:
            out_dict["dimensions_synt"].append([output, sample])
        else:
            out_dict["dimensions_sem"].append([output, sample])            

scores = dict()
for task in tasks:
    print(task, end=':')
    scores[task] = 0
    if task=="dimensions_sem":
        scores[task] = [0, 0]
    true_cnt = 0 if ("dimensions" in task or "volume" in task) else len(out_dict[task])
    for output, sample in out_dict[task]:
        if task=="atom count":
            output = output.strip().replace('\n', '').replace(' ', '')
            if output.isdigit():
                scores[task] += eval_atom_cnt(sample, int(output))
        if task=="dimensions_synt":
            l = output.split(',')
            l = [x for x in l if x!='']
            l = list(map(lambda x: x.strip().replace('\n', '').replace(' ', ''), l))
            if len(l)==3:
                try:
                    scores[task] += eval_dimensions(sample, [float(l[0]), float(l[1]),float(l[2])])
                    true_cnt += 1
                except:
                    continue
        if task=="atom name":
            output = output.strip().replace('\n', '').replace(' ', '')
            scores[task] += eval_atom_name(sample, output)
        if task=="replace":
            scores[task] += eval_replace(sample, output)
        if task=="space group":
            output = output.strip().replace('\n', '').replace(' ', '')
            scores[task] += eval_spacegroup(sample, output)
        if task=="cell_volume":
            output = output.strip().replace('\n', '').replace(' ', '')
            try:
                answer = float(output)
                scores[task] += eval_cell_volume(sample, answer)
                true_cnt += 1
            except:
                continue
        if task=="formula":
            output = output.strip().replace('\n', '').replace(' ', '')
            scores[task] += eval_formula(sample, output)
        if task=="infill":
            output = output.strip().replace('\n', '').replace(' ', '')
            scores[task] += eval_infill_task(sample, output)
        if task=="dimensions_sem":
            l1 = output.split('\n')
            l1 = [x for x in l1 if x!='']
            if len(l1)!=2:
                continue
            l21 = l1[0].split()
            l22 = l1[1].split()
            l21 = [x.strip() for x in l21 if x!='']
            l22 = [x.strip() for x in l22 if x!='']
            if len(l21)==3 and len(l22)==3:
                mse, mae = eval_dimensions_sem(sample, [[float(l21[0]), float(l21[1]),float(l21[2])], [float(l22[0]), float(l22[1]), float(l22[2])]])
                true_cnt += 1
                scores[task][0] += mse
                scores[task][1] += mae
        if task=="vol_calc":
            output = output.strip().replace('\n', '').replace(' ', '')
            try:
                answer = float(output)
                scores[task] += eval_cell_volume(sample, answer)
                true_cnt += 1
            except:
                continue 
        if task=="formula_compute":
            output = output.strip().replace('\n', '').replace(' ', '')
            scores[task] += eval_formula(sample, output)
        if "generation" in task:
            scores[task] += eval_gen_format(output)
        
    if "dimensions_sem" not in task:
        scores[task] /= true_cnt
    else:
        scores[task][0] /= true_cnt
        scores[task][1] /= true_cnt

    print(scores[task], true_cnt, len(out_dict[task]))

with open('results.txt', 'w') as f:
    for task in out_dict:
        f.write(task + ": " + str(scores[task]) + '\n')

with open('samples.txt', 'w') as f:
    for task in out_dict:
        f.write(task + ":" + '\n')
        f.write(out_dict[task][0][1]['system']+out_dict[task][0][1]['input'] +'\n')
        f.write("Expected output:" +'\n')
        f.write(str(out_dict[task][0][1]['output']) +'\n')
        f.write("Model output:" +'\n')
        f.write(out_dict[task][0][0] +'\n')