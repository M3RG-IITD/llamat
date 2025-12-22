# %%
import pickle
import numpy as np
import pandas as pd
import sys

curtest = sys.argv[1] + '_mof1_test.pkl';  
# if('basic' in curtest):
#     curname = curtest.split('.')[0];
# elif('llama2' in curtest):
#     curname = sys.argv[1];
# elif('orca' in curtest):
#     curname = '_'.join(sys.argv[1].split('_')[:5]);
# else:
#     curname = '_'.join(sys.argv[1].split('_')[:3]);

curname = sys.argv[1];

with open(curtest,'rb') as f:
    outputcs = pickle.load(f)
f.close()

split_token = '<|im_e'
if('llama2' in curtest or 'llamat2' in curtest):
    split_token = '\n an';
print("split_token: ", split_token);
# %%
# %% [markdown]
# <h1>eval</h1>

# %%
import random
import json
def load_jsonl(path):
    with open(path, 'r') as f:
        a = f.readlines()
        g = [json.loads(i) for i in a]
    return g

# %%
mof1_train = load_jsonl('mof1_train.jsonl')
mof1_test = load_jsonl('mof1_test.jsonl')
results_ = {};
import json
import os
import numpy as np
import jellyfish
import copy
import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
import pprint

def check_equivalence_of_entries(gold_entry, test_entry):
    ## Entries are a list of dictionaries
    ## We first order each list, then each dictionary, then compare strings
    ### order list by formula key
    gold_entry = sorted(gold_entry, key=lambda x: x.get('formula', ''))
    test_entry = sorted(test_entry, key=lambda x: x.get('formula', ''))

    ### order each dictionary by keys
    gold_entry = [dict(sorted(d.items())) for d in gold_entry]
    test_entry = [dict(sorted(d.items())) for d in test_entry]

    ### compare strings
    return str(gold_entry) == str(test_entry)

# %%
import ast
consider = []
for sid, s in enumerate(mof1_test):
    if ast.literal_eval(s['answer']) != []:
        consider.append(sid)

# from utils import read_jsonl
def ent_str_to_words(ent):
    stripped =  [e.strip() for e in ent.split(" ")]
    return [e for e in stripped if e]


def ent_json_to_word_basis_sets(ent_json, return_empty=False):
    """
    Where ent_json is multiple entries in a list

    Return all entities and links in a set-based word basis
    """
    # Must account for these in a weird way because the entries are not ordered :(
    to_account = {e: set() for e in ENTS_FROZEN + ENTS_LINKS_FROZEN}

    # for purposes of counting support only
    to_account_aux_ents_only = {e: set() for e in ENTS_FROZEN + ENTS_LINKS_FROZEN}

    if return_empty:
        return to_account, {}

    for entry in ent_json:
        root_accounting = {root: set() for root in ROOT}
        for etype in ENTS_FROZEN:
            ent_strs = entry[etype]
            if isinstance(ent_strs, str) and ent_strs:
                to_account_aux_ents_only[etype].add(ent_strs)
                for w in ent_str_to_words(ent_strs):
                    to_account[etype].add(w)
                if etype in ROOT and ent_strs:
                    # Formulae/roots must be counted as single words
                    root_accounting[etype].add(ent_strs)
                    # root_accounting[etype] = root_accounting[etype].union(set(ent_str_to_words(ent_strs)))
            elif isinstance(ent_strs, list):
                for ent_str in ent_strs:
                    if ent_str:
                        to_account_aux_ents_only[etype].add(ent_str)
                        for w in ent_str_to_words(ent_str):
                            if w:
                                to_account[etype].add(w)
            elif ent_strs:
                raise ValueError(f"Ent strings was a weird type: {type(ent_strs)}, {ent_strs}")

        # Add links
        for root, accounting in root_accounting.items():
            if accounting:
                for e in ENTS_FROZEN_NOROOT:
                    ent_strs = entry[e]
                    words = []
                    if isinstance(ent_strs, str):
                        words = ent_str_to_words(ent_strs)

                    elif isinstance(ent_strs, list):
                        for ent_str in ent_strs:
                            words += ent_str_to_words(ent_str)
                    else:
                        raise ValueError(f"Ent strings was a weird type: {type(ent_strs)}, {ent_strs}")

                    if words:
                        for f in accounting:
                            for w in words:
                                # avoid self-links
                                if f != w:
                                    to_account[f"{root}{LINK_DELIMITER}{e}"].add(f"{f}{LINK_DELIMITER}{w}")

                            if isinstance(ent_strs, str):
                                to_account_aux_ents_only[f"{root}{LINK_DELIMITER}{e}"].add(f"{f}{LINK_DELIMITER}{ent_strs}")
                            else:
                                for ent_str in ent_strs:
                                    to_account_aux_ents_only[f"{root}{LINK_DELIMITER}{e}"].add(f"{f}{LINK_DELIMITER}{ent_str}")
    return to_account, to_account_aux_ents_only

printmode = False;
all_results = []
all_winkler_similarities = []
all_exact_match_accuracy = []
all_unparsable = []


TASK = "general"

if TASK == "mof":
    ENTS_FROZEN = ['name_of_mof', 'mof_formula', 'mof_description', 'guest_species', 'applications']
elif TASK == "general":
    ENTS_FROZEN = ["acronym", "applications", "name", "formula", "structure_or_phase", "description"]
    # ENTS_FROZEN = ["applications", "name", "formula", "structure_or_phase", "description"]
LINK_DELIMITER = "|||"
if TASK == "mof":
    ROOT = ("name_of_mof",)
elif TASK == "general":
    ROOT = ("formula",)
else:
    raise ValueError(f"There is no task '{TASK}'")
    
ENTS_FROZEN_NOROOT = [e for e in ENTS_FROZEN if e not in ROOT]
ENTS_LINKS_FROZEN = [f"{root}{LINK_DELIMITER}{e}" for e in ENTS_FROZEN_NOROOT for root in ROOT]


support = {
    "ents": {e: 0 for e in ENTS_FROZEN},
    "words": {e: 0 for e in ENTS_FROZEN},
    "links_ents": {e: 0 for e in ENTS_LINKS_FROZEN},
    "links_words": {e: 0 for e in ENTS_LINKS_FROZEN}
}

tn = []
prompts = []
for m in mof1_test:
    if m['question'].startswith('Here'):
        tn.append(len(m['question'].split('\n\n')[1].strip().split()))
        prompts.append(m['question'].split('\n\n')[1].strip())
    else :
        tn.append(len(m['question'].split('\n\n')[0].strip().split()))
        prompts.append(m['question'].split('\n\n')[0].strip())

# %%
len(consider), len(mof1_test), len(prompts)

# %%
sss = {"prompt": "Applications of metal-organic frameworks as stationary phases in chromatography\nAs a new family of mesoporous and microporous materials, metal-organic frameworks (MOFs) are considered versatile materials for widespread technical applications. We summarize some of the eminent properties of MOFs and the development in the application of MOFs in chromatography, especially the advantages they could bring to gas chromatography (GC) and high-performance liquid chromatography (HPLC). The application of MOFs as novel stationary phases in chromatography in place of conventional materials has led to notable improvements in the performance of GC and HPLC. We highlight differences in chromatographic performance between MOFs and conventional sorbents, such as zeolites, before discussing the future prospects for MOFs in chromatography.\n\n###\n\n", "completion": " [{\"name_of_mof\": \"\", \"mof_formula\": \"\", \"mof_description\": [\"\"], \"guest_species\": [\"\"], \"applications\": [\"chromatography\"]}]\n\nEND\n\n", "llm_completion": "[{\"name_of_mof\": \"\", \"mof_formula\": \"\", \"mof_description\": [\"\"], \"guest_species\": [\"\"], \"applications\": [\"chromatography\", \"gas chromatography\", \"high-performance liquid chromatography\"]}]"}
sss['completion']


# %%
run  = []; missed = [];
total = 0; failed = 0;
all_total_cnt = 0;
for fn in consider:
    total += 1;
    all_total_cnt += 1;
    sample = dict()
    sample['completion'] = json.loads(mof1_test[fn]['answer'])
    try:
        line=(outputcs[fn].split(split_token)[0].strip()).replace(',',',')
        line = line.replace("'", '"') #Since Json accepts only " inside. although this is probably not the issue.
        sample['llm_completion'] = json.loads(line)
    except Exception as e:
        print("missed decoding due to ", e)
        print("LINE:\n");
        print(line);
        print();
        print("Actual:", sample['completion'])
        failed += 1; 
        missed.append(fn);
        sample['llm_completion'] = [];
        continue;
        # print(f"failed at fn= {fn}, path={outputcs[fn].split(split_token)[0].strip()}");
        # raise e;
    sample['prompt'] = prompts[fn]
    run.append(sample)
print("missed:", missed)

# %%
len(missed)

# %%
printmode = False;

# %%
for filenum in range(1):
    # run = read_jsonl(os.path.join(RESULTS_DIR, fn))
    exact_matches = 0
    unparsable = 0
    total = 0
    jaro_winkler_similarities = []
    ent_scores_test = {e: [] for e in ENTS_FROZEN}
    ent_scores_gold = {e: [] for e in ENTS_FROZEN}
    subdict = {"test_correct_triplets": 0, "test_retrieved_triplets": 0, "gold_retrieved_triplets": 0}
    links_scores = {el: copy.deepcopy(subdict) for el in ENTS_LINKS_FROZEN}

    for ie, sample in tqdm.tqdm(enumerate(run)):
        gold_string = str(sample["completion"]) #.replace("\n\nEND\n\n", "").strip()
        test_string = str(sample["llm_completion"]) #.replace("\n\nEND\n\n", "").replace('\\', '').strip()

        # gold_json = json.loads(gold_string)
        gold_json = sample["completion"]
        prompt = sample["prompt"] ##.replace("\n\n###\n\n", "").strip()
        n_prompt_words = len([w for w in prompt.split(" ") if w])

        total += 1
        # if gold_string == test_string:
        #     exact_matches += 1
        test_json = {}
        was_unparsable = False
        try:
            test_json = sample["llm_completion"]
            # print("test_json is", test_json);
            if(test_json == []):
                json.loads('(])'); #raises a jsonDecodeError.
            if isinstance(test_json, dict):
                test_json = [test_json]
            if isinstance(test_json, str):
                raise Exception("found test_json taht is string: ", test_json);
                try:
                    test_json = sample["llm_completion"]#json.loads(test_json)
                    # test_json = json.loads(test_json)
                except json.decoder.JSONDecodeError as jse:
                    test_json = []
            for d in test_json:
                for key in ENTS_FROZEN:
                    if key not in d:
                        if key in ["formula", "name", "acronym", "mof_formula", "name_of_mof"]:
                            try:
                                d[key] = ""
                            except Exception as e:
                                print("d:", d, " and type:" ,type(d));
                                print("error raised: e", e);
                                print("*"*20);
                                print("d[key]:", d[key], " and type: ", type(d[key]));
                                raise e;
                        else:
                            d[key] = [""]

                # remove extra keys as they are "parsable" but invalid
                extra_keys = []
                for key in d:
                    if key not in ENTS_FROZEN:
                        extra_keys.append(key)
                for key in extra_keys:
                    d.pop(key)

        except json.decoder.JSONDecodeError as jse:
            unparsable += 1
            was_unparsable = True
        
        if check_equivalence_of_entries(gold_json, test_json):
            exact_matches += 1
            was_exact = True
        else:
            was_exact = False

        jws = jellyfish.jaro_winkler_similarity(gold_string, test_string, long_tolerance=True)
        jaro_winkler_similarities.append(jws)

        gold_accounting, gold_accounting_support_helper = ent_json_to_word_basis_sets(gold_json)

        if test_json:
            test_accounting, _ = ent_json_to_word_basis_sets(test_json)
        else:
            test_accounting, _ = ent_json_to_word_basis_sets({}, return_empty=True)

        # this loop is used only for collecting numbers for support
        # of both multiword ents and the number of words (for both NER and relational)

        for k, v in gold_accounting_support_helper.items():
            if LINK_DELIMITER in k:
                support["links_ents"][k] += len(set(v))
                support["links_words"][k] += len(gold_accounting[k])
            else:
                support["ents"][k] += len(set(v))
                support["words"][k] += len(gold_accounting[k])

        if printmode:
            print(f"Entry {ie+1} of {len(run)} samples of file {fn}")
            print(f"Gold entry was {gold_json}")
            print(f"Test string is {test_json}")
            print(f"Was exact match: {was_exact}")
            print(f"Was unparsable: {was_unparsable}")

        for etype in ENTS_FROZEN:
            ent_accounting_copy = copy.deepcopy(test_accounting[etype])
            n_unlabelled_words = copy.deepcopy(n_prompt_words)
            for ew in gold_accounting[etype]:

                # Account for true positives
                if ew in test_accounting[etype]:
                    ent_scores_test[etype].append(1)
                    ent_scores_gold[etype].append(1)
                    ent_accounting_copy.remove(ew)
                    n_unlabelled_words -= 1
                # account for false negatives
                else:
                    ent_scores_test[etype].append(0)
                    ent_scores_gold[etype].append(1)
                    n_unlabelled_words -= 1

            # Among the remaining test accounting words, only false positives
            # should remain in the set
            for ew in ent_accounting_copy:
                ent_scores_test[etype].append(1)
                ent_scores_gold[etype].append(0)
                n_unlabelled_words -= 1

            # the only labels remaining are true negatives
            ent_scores_test[etype] += [0] * n_unlabelled_words
            ent_scores_gold[etype] += [0] * n_unlabelled_words

        for elinktype in ENTS_LINKS_FROZEN:
            gold_triples = gold_accounting[elinktype]
            test_triples = test_accounting[elinktype]

            correct_triples = [e for e in test_triples if e in gold_triples]
            n_correct_triples = len(correct_triples)
            links_scores[elinktype]["test_correct_triplets"] += n_correct_triples
            links_scores[elinktype]["test_retrieved_triplets"] += len(test_triples)
            links_scores[elinktype]["gold_retrieved_triplets"] += len(gold_triples)

            if printmode:
                print(f"\tLink type: {elinktype}")
                print(f"\t\tTrue positives ({len(correct_triples)}: {pprint.pformat(correct_triples)}")
                false_negatives = [e for e in gold_triples if e not in test_triples]
                false_positives = [e for e in test_triples if e not in gold_triples]
                print(f"\t\tFalse negatives ({len(false_negatives)}): {pprint.pformat(false_negatives)}")
                print(f"\t\tFalse positives({len(false_positives)})= {pprint.pformat(false_positives)}")
        if printmode:
            print("-"*30)


    results = {"ents": {}, "links": {}}
    for etype in ENTS_FROZEN:
        gold_arr = ent_scores_gold[etype]
        test_arr = ent_scores_test[etype]

        subdict = {"recall": 0, "precision": 0, "f1": 0}
        subdict["recall"] = recall_score(gold_arr, test_arr)
        subdict["precision"] = precision_score(gold_arr, test_arr)
        subdict["f1"] = f1_score(gold_arr, test_arr)
        results["ents"][etype] = subdict


    for elinktype in ENTS_LINKS_FROZEN:
        subdict = {} #"precision": 0, "recall": 0, "f1": 0}
        n_correct = links_scores[elinktype]["test_correct_triplets"]
        n_retrieved = links_scores[elinktype]["test_retrieved_triplets"]
        n_gold_retrieved = links_scores[elinktype]["gold_retrieved_triplets"]

        try:
            subdict["precision"] = n_correct/n_retrieved
            subdict["recall"] = n_correct/n_gold_retrieved
        except ZeroDivisionError: # if n_retrieved or n_gold_retrieved is zero, do not append this fold
            results["links"][elinktype] = {}#subdict # {}
            continue
        if n_correct == 0: # equivalent to subdict["precision"]==0 & subdict["recall"]==0
            subdict["f1"] = 0 # not actually defined but at least this is strict
        else:
            subdict["f1"] = 2 * (subdict["precision"] * subdict["recall"])/(subdict["precision"] + subdict["recall"])
        results["links"][elinktype] = subdict

    all_exact_match_accuracy.append(exact_matches/total)
    all_winkler_similarities.append(np.mean(jaro_winkler_similarities))
    all_unparsable.append(unparsable/total)
    all_results.append(results)
    results_[curname] = results;
    count_parsable = (len(run) - unparsable, all_total_cnt);


print("Summary: \n" + "-"*20)
print("Support was ", pprint.pformat(support))
print("All Exact match accuracy average:", np.mean(all_exact_match_accuracy))
print("Jaro-Winkler avg similarity:", np.mean(all_winkler_similarities))
print("Parsable percentage", 1-np.mean(all_unparsable))
print("parsable ratio: ", len(mof1_test) - np.sum(all_unparsable) , "/", len(mof1_test))
print(all_unparsable);
print("parsable ratio: ", count_parsable[0] , "/", count_parsable[1])
results_[curname]['parsable'] = count_parsable;

# %%
outer_keys = ("links", "ents")
inner_keys = ("recall", "precision", "f1")

if printmode:
    print("Results by fold:")
    pprint.pprint(all_results)

r_dict_avg = copy.deepcopy(all_results[0])
for k, v in r_dict_avg.items():
    try:
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                r_dict_avg[k][k2][k3] = None
    except:
        continue;

for ok in outer_keys:
    if ok == "links":
        mid_keys = ENTS_LINKS_FROZEN
    else:
        mid_keys = ENTS_FROZEN
    for mk in mid_keys: # elink
        for ik in inner_keys: # recall/precision/f1
            arr2avg = []
            for foldix, rd in enumerate(all_results): # fold
                if rd[ok][mk]=={}: # pass for this fold
                    print("skipped", ok, mk, ik, "for fold", foldix, "due to insufficient gold data for link")
                    continue
                else:
                    arr2avg.append(rd[ok][mk][ik])
            if printmode:
                print(f"For {ok}-{mk}-{ik} we find {arr2avg} -> {np.mean(arr2avg)}")
            r_dict_avg[ok][mk][ik] = np.mean(arr2avg) #average over folds

pprint.pprint(r_dict_avg)

# %%
cnt = 0; 
for k in results_.keys():
    if cnt == 0:
        print("model, ", end = "");
        for j in results_[k]['ents'].keys():
            print(j, end = ", ");
        print("parsable");
        print();
    cnt += 1;
    print(k, ",", end ="");
    for j in results_[k]['ents'].keys():
        print(round(results_[k]['ents'][j]['f1'], 3), end = ", ");
    print(f"{results_[k]['parsable'][0]}/{results_[k]['parsable'][1]}");
    print();    

# %%
# results_['llama2']['links'].keys()
print(*["*" for _ in range(20)]);
print("entity-links:");
# %%
#consider = ['formula|||applications', 'formula|||description', 'formula|||structure_or_phase']
# consider = ['formula|||name','formula|||acronym',  'formula|||structure_or_phase', 'formula|||applications', 'formula|||description']

consider = ['formula|||name',  'formula|||structure_or_phase', 'formula|||applications', 'formula|||description']

for j in consider:
    print(j, end = ", ");
    print(round(results_[k]['links'][j]['precision'], 3), end = ", ");
    print(round(results_[k]['links'][j]['recall'], 3), end = ", ");
    print(round(results_[k]['links'][j]['f1'], 3), end = ", ");
    print();

cnt = 0; 
#for k in results_.keys():
#    if cnt == 0:
#        print("model, ", end = "");
#        for j in consider:
#            print(j, end = ", ");
#        print("parsable");
#        print();
#    cnt += 1;
#    print(k, ",", end ="");
#    # for j in results_[k]['links'].keys():
#    # 
#    for j in consider:
#        print(j, end = ", ");
#        print(round(results_[k]['links'][j]['precision'], 3), end = ", ");
#        print(round(results_[k]['links'][j]['recall'], 3), end = ", ");
#        print(round(results_[k]['links'][j]['f1'], 3), end = ", ");
#        print();
#    print(f"{results_[k]['parsable'][0]}/{results_[k]['parsable'][1]}");
#    print();    

# %%


# %%

