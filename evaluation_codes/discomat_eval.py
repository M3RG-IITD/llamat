import ast
import numpy as np
import pickle
import os
import json
import pandas as pd
import random
import sys
results_ = {};
if(len(sys.argv) < 2):
    print("inputsize = ",len(sys.argv) ,"format = python3 discomat_eval.py $1 $2, where $1 = pickle file, $2 = name (optional)")
    exit();


arg1 = sys.argv[1]
suff = "_discomat_test.pkl"
if arg1[-len(suff):] != suff:
    arg1 = sys.argv[1] + suff;

namesplit = arg1.split('_');
if(len(sys.argv) < 3):
	if("llama_basic" in arg1):
		arg2 = arg1;
	elif("orca" not in arg1):
		arg2 = "_".join(namesplit[:3]);
	else:
		arg2 = "_".join(namesplit[:5]);
else:
	arg2 = sys.argv[2];

curname = arg2;
curtest = arg1; #  tests[idx];
with open(curtest,'rb') as f:
    outputcs = pickle.load(f)
f.close()
results_[curname] = {}; 
results_[curname]['ds'] = {}


if 'llama2' in curtest or 'llamat2' in curtest : split_token = '\n an'
else: split_token = '<|im_e'

## WHEN USING LLAMAT MODELS, use '\n an' as split token for LLaMat-2 and '<|im_e' as split token for LLaMat-3

valfilepath = 'discomat_test_check.pkl'
with open(valfilepath,'rb') as f:
    valfile = pickle.load(f)
f.close()

def format_discomat_preds(idx):
    disco_pred_answer = dict()
    rcd = valfile[idx]['pii_tid']['discomat']['pred_row_col_labels']
    
    # for tables having chemical names in rows, and one complete comp is in 1 column
    if 2 in rcd['row']:
        disco_pred_answer['chemical_row_index'] = list(np.where(np.array(rcd['row']) == 2)[0])
        disco_pred_answer['composition_col_index'] = list(np.where(np.array(rcd['col']) == 1)[0])
        
        if 3 in rcd['row']:
            disco_pred_answer['gid_row_index'] = [rcd['row'].index(3)]
    
    if 2 in rcd['col']:
        disco_pred_answer['chemical_col_index'] = list(np.where(np.array(rcd['col']) == 2)[0])
        disco_pred_answer['composition_row_index'] = list(np.where(np.array(rcd['row']) == 1)[0])
        
        if 3 in rcd['col']:
            disco_pred_answer['gid_col_index'] = [rcd['col'].index(3)]
    
    if valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] == 0:
        disco_pred_answer['comp_table'] = [1]
        disco_pred_answer['regex_table'] = [1]
        if 3 in rcd['row']:
            disco_pred_answer['gid_row_index'] = [rcd['row'].index(3)]
        if 3 in rcd['col']:
            disco_pred_answer['gid_col_index'] = [rcd['col'].index(3)]
        
    if 0 < valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] < 3:
        disco_pred_answer['comp_table'] = [1]
        disco_pred_answer['regex_table'] = [0]
    
    if valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] == 3:
        disco_pred_answer['comp_table'] = [0]
        disco_pred_answer['regex_table'] = [0]
    
    return disco_pred_answer

exact_match = {'discomat':0, 'llamat2':0}
missed = []
for i in range(len(valfile)):
    actual = valfile[i]['answer']
    pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
    if outputcs[i] != 2048:
        try:
            # pred_2 = outputcs[i]
            # pred = ast.literal_eval(pred_2.split('\n an')[0])
            if pred_d == actual:
                exact_match['discomat'] += 1
            # elif pred == actual:
            #     exact_match['llamat2'] += 1
        except:
            missed.append(i)
missed = []

for i in range(len(valfile)):
    actual = valfile[i]['answer']
    pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
    if outputcs[i] != 2048:
        try:
            pred_2 = outputcs[i]
            pred = ast.literal_eval(pred_2.split(split_token)[0])
            # if pred_d == actual:
            #     exact_match['discomat'] += 1
            if pred == actual:
                exact_match['llamat2'] += 1
        except:
            missed.append(i)
    

print(f"missed {len(missed)} out of {len(valfile)}");

print("MISSED:", missed);
print(*["*" for i in range(40)]);
longlen = 0
for t in outputcs:
    if t == 2048:longlen+=1
print(f'Tables non parsable = {len(missed)}/{len(valfile)-longlen}')
for k,v in exact_match.items():
    print(k)
    print(f'Exact match for {k}:', v)
    
    if k=='discomat':
        print(f'Fraction Exact match for {k}:', v/(len(valfile)))
#         results_['discomat']['exact_match'] =  [v,(len(valfile))]
        results_[curname]['ds']['exact_match'] = [v,(len(valfile))]; 
    else:
        print(f'Fraction Exact match for {k}:', v/(len(valfile)-len(missed)-longlen))
        results_[curname]['exact_match'] = [v,(len(valfile)-len(missed)-longlen)];
    # print(f'Exact match for {k}:', v)

act_comptable = []
act_regextable = []
pred_comptable_discomat = []
pred_comptable_llamat = []
pred_regex_discomat = []
pred_regex_llamat = []


# for discomat
for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            actual = valfile[i]['answer']
            act_comptable.append(actual['comp_table'][0])
            act_regextable.append(actual['regex_table'][0])
            
            pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
            pred_comptable_discomat.append(pred_d['comp_table'][0])
            pred_regex_discomat.append(pred_d['regex_table'][0])
        
    
            pred_2 = outputcs[i]
            pred = ast.literal_eval(pred_2.split(split_token)[0])
            
            try:
                pred_comptable_llamat.append(pred['comp_table'][0])
            except:
                pred_comptable_llamat.append(0)
            
            try:
                pred_regex_llamat.append(pred['regex_table'][0])
            except:
                pred_regex_llamat.append(0)


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

target = ['NON-C','COMP']
print('discomat')
print('Report', classification_report(act_comptable, pred_comptable_discomat, target_names=target))

print('llamat')
print('Report', classification_report(act_comptable, pred_comptable_llamat, target_names=target))

results_[curname]['comptable'] = [(f1_score(act_comptable, pred_comptable_llamat)), np.array(act_comptable).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['comptable'] = (f1_score(act_comptable, pred_comptable_discomat)); 


from sklearn.metrics import classification_report

target = ['no-regex','regex']
print('discomat')
print('Report', classification_report(act_regextable, pred_regex_discomat, target_names=target))

print('llamat')
print('Report', classification_report(act_regextable, pred_regex_llamat, target_names=target))


results_[curname]['regex'] = [(f1_score(act_regextable, pred_regex_llamat)), np.array(act_regextable).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['regex'] = (f1_score(act_regextable, pred_regex_discomat)); 


gid_actual = []
gid_pred = []

gid_discomat_actual = []
gid_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'gid_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # gid_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['gid_row_index']
                
                
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    # gid_discomat_actual.append(1)
                    gid_actual.append(1)
                    
                    if 'gid_row_index' not in pred.keys():
                        pred['gid_row_index'] = []


                    if 'gid_row_index' not in pred_d.keys():
                        pred_d['gid_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if pred['gid_row_index'] == actual_label:
                        gid_pred.append(1)
                    else:
                        gid_pred.append(0)


                    if pred_d['gid_row_index'] == actual_label:
                        gid_discomat_pred.append(1)
                    else:
                        gid_discomat_pred.append(0)
                # else:
                #     gid_pred.append(0)
                    
            if 'gid_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # gid_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['gid_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    gid_actual.append(1)

                    try:
                        pred.keys();
                    except Exception as e:
                        # print(e);
                        print("PRED IS  OF TYPE: ", type(pred))
                        print(pred);
                        raise e;
                    # gid_discomat_actual.append(1)
                    if 'gid_col_index' not in pred.keys():
                        pred['gid_col_index'] = []

                    if 'gid_col_index' not in pred_d.keys():
                        pred_d['gid_col_index'] = []
                    # print(actual_label, pred['gid_col_index'])

                    if pred['gid_col_index'] == actual_label:
                        gid_pred.append(1)
                    else:
                        gid_pred.append(0)

                    if pred_d['gid_col_index'] == actual_label:
                        gid_discomat_pred.append(1)
                    else:
                        gid_discomat_pred.append(0)
                # else:
                #     gid_pred.append(0)
                

from sklearn.metrics import f1_score, recall_score, precision_score


# Glass id scores
from sklearn.metrics import classification_report

print('discomat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(gid_actual, gid_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(gid_actual, gid_pred),3))

# 737-144-4
results_[curname]['gid'] = [(f1_score(gid_actual, gid_pred)), np.array(gid_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['gid'] = (f1_score(gid_actual, gid_discomat_pred)); 


composition_actual = []
composition_pred = []

composition_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'composition_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['composition_row_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    
                    if 'composition_row_index' not in pred.keys():
                        pred['composition_row_index'] = []


                    if 'composition_row_index' not in pred_d.keys():
                        pred_d['composition_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['composition_row_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    if sorted(pred_d['composition_row_index']) == sorted(actual_label):
                        composition_discomat_pred.append(1)
                    else:
                        composition_discomat_pred.append(0)
                    
            if 'composition_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['composition_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    
                    if 'composition_col_index' not in pred.keys():
                        pred['composition_col_index'] = []


                    if 'composition_col_index' not in pred_d.keys():
                        pred_d['composition_col_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['composition_col_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    if sorted(pred_d['composition_col_index']) == sorted(actual_label):
                        composition_discomat_pred.append(1)
                    else:
                        composition_discomat_pred.append(0)


# composition row-col scores
from sklearn.metrics import classification_report

print('discomat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_pred),3))

# 737-144-4
results_[curname]['composition'] = [(f1_score(composition_actual, composition_pred)), np.array(composition_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['composition'] = (f1_score(composition_actual, composition_discomat_pred)); 


composition_actual = []
composition_pred = []

composition_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'chemical_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['chemical_row_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    if 'chemical_row_index' not in pred.keys():
                        pred['chemical_row_index'] = []


                    if 'chemical_row_index' not in pred_d.keys():
                        pred_d['chemical_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['chemical_row_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    if sorted(pred_d['chemical_row_index']) == sorted(actual_label):
                        composition_discomat_pred.append(1)
                    else:
                        composition_discomat_pred.append(0)
                    
            if 'chemical_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                pred_d = format_discomat_preds(i)
                
                actual_label = actual['chemical_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    if 'chemical_col_index' not in pred.keys():
                        pred['chemical_col_index'] = []


                    if 'chemical_col_index' not in pred_d.keys():
                        pred_d['chemical_col_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['chemical_col_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    if sorted(pred_d['chemical_col_index']) == sorted(actual_label):
                        composition_discomat_pred.append(1)
                    else:
                        composition_discomat_pred.append(0)
                

# chemical row-col scores
from sklearn.metrics import classification_report

print('discomat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_pred),3))


results_[curname]['chemical'] = [(f1_score(composition_actual, composition_pred)), np.array(composition_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['chemical'] = (f1_score(composition_actual, composition_discomat_pred)); 


print('discomat')
print('Precision', precision_score(act_comptable, pred_comptable_discomat,labels=[1,0]))

print('llamat')
print('Precision', precision_score(act_comptable, pred_comptable_llamat))

print('discomat')
print('Reccall', recall_score(act_comptable, pred_comptable_discomat))

print('llamat')
print('Reccall', recall_score(act_comptable, pred_comptable_llamat))


print('discomat')
print('F1-score', f1_score(act_comptable, pred_comptable_discomat))

print('llamat')
print('F1-score', f1_score(act_comptable, pred_comptable_llamat))


def printR(*args, end = "\n"):
    for x in args:
        if(type(x) == float or type(x) == np.float64 or type(x) == np.float32):
            print(round(x,3), end = "");
        else:
            print(x, end = "");
    print(end, end = "");
    
cur = 0;
for model in results_.keys():
    
    if(cur == 0):
        print("model, ", end = "");
        for task in results_[model].keys():
            if(task == 'ds'): 
                continue; 
            if(task != 'exact_match'):
                print(task, ",ds", ",support", end = ",");
            else:
                print(task, ",ds", end = ",");
        print();
    cur += 1;
    printR(model, end = ",")
    for task in results_[model].keys():
        if(task == 'ds'): 
            continue; 
        if(task != 'exact_match'):
            printR(results_[model][task][0], ",", results_[model]['ds'][task], ",", results_[model][task][1], end = ",");
        else:
            printR(results_[model][task][0],'/',results_[model][task][1], ",", results_[model]['ds'][task][0],"/", results_[model]['ds'][task][1], end = ",");

    print()
