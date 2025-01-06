import pickle
# %%
import jellyfish
import random
import sys
import json
def load_jsonl(path):
    with open(path, 'r') as f:
        a = f.readlines()
        g = [json.loads(i) for i in a]
    return g

# %%
mof1_train = load_jsonl('mof2_train.jsonl')
mof1_test = load_jsonl('mof2_test.jsonl')

# %%
import ast

# %%
import pickle

# %%
current = sys.argv[1] + "_mof2_test.pkl"
split_token = '<|im_end|>'; ## FOR LLAMA3 Generations

if('llama2' in current or 'llamat2' in current):
    split_token = '\n an';  ## FOR LLAMA2 Generations
    print("CHANGED SPLIT_TOKEN");

with open(current,'rb') as f:
    outputcs = pickle.load(f)
f.close()

print(f"model output: {current}, split token: {split_token}");

# %%
import json
import os
import numpy as np
import jellyfish
import copy
import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
import pprint

# %%
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

# %%
consider = []
for sid, s in enumerate(mof1_test):
    if ast.literal_eval(s['answer']) != []:
        consider.append(sid)
    # print(s['answer'])

# %%
len(consider), len(mof1_test)

# %%


# %%
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


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--results_dir", type=str, default="predictions_general_gpt3")
#     # must be general or mof
#     parser.add_argument("--task", type=str, default="general", choices=["general", "mof"], help="Which schema is being used")

#     parser.add_argument(
#         "--loud",
#         action='store_true',
#         help="If true, show a summary of each evaluated sentence w/ FP and FNs.",
#         required=False
#     )

#     args = parser.parse_args()
#     printmode = args.loud

#     RESULTS_DIR = args.results_dir
#     TASK = args.task


all_results = []
all_winkler_similarities = []
all_exact_match_accuracy = []
all_unparsable = []


TASK = "mof"

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




# %%
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
len(tn), len(prompts)

# %%
run  = []; missed = [];
for fn in consider:
    sample = dict()
    sample['completion'] = json.loads(mof1_test[fn]['answer'])
    try:
        sample['llm_completion'] = json.loads(outputcs[fn].split(split_token)[0])
    except:
        missed.append(fn);
        sample['llm_completion'] = [];
    sample['prompt'] = prompts[fn]
    run.append(sample)
print("missed:", missed)

# %%
printmode = False; #too many prints otherwise

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
            if(test_json == []):
                json.loads('(('); ## TO raise json.decoder.JSONDecodeError.
            if isinstance(test_json, str):
                try:
                    test_json = sample["llm_completion"]#json.loads(test_json)
                except json.decoder.JSONDecodeError as jse:
                    test_json = []
            for d in test_json:
                for key in ENTS_FROZEN:
                    if key not in d:
                        if key in ["formula", "name", "acronym", "mof_formula", "name_of_mof"]:
                            d[key] = ""
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

print("Summary: \n" + "-"*20)
print("Support was ", pprint.pformat(support))
print("All Exact match accuracy average:", np.mean(all_exact_match_accuracy))
print("Jaro-Winkler avg similarity:", np.mean(all_winkler_similarities))
print("Parsable percentage", 1-np.mean(all_unparsable))

# %%
outer_keys = ("links", "ents")
inner_keys = ("recall", "precision", "f1")

if printmode:
    print("Results by fold:")
    pprint.pprint(all_results)

r_dict_avg = copy.deepcopy(all_results[0])
for k, v in r_dict_avg.items():
    for k2, v2 in v.items():
        for k3, v3 in v2.items():
            r_dict_avg[k][k2][k3] = None


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
## Now to store the results and print in csv format
final_results = {'13k': {'ents': {'name_of_mof': {'recall': 0.7164179104477612,
    'precision': 0.7619047619047619,
    'f1': 0.7384615384615385},
   'mof_formula': {'recall': 0.32142857142857145,
    'precision': 0.8181818181818182,
    'f1': 0.46153846153846156},
   'mof_description': {'recall': 0.2597402597402597,
    'precision': 0.47619047619047616,
    'f1': 0.33613445378151263},
   'guest_species': {'recall': 0.3076923076923077,
    'precision': 0.7272727272727273,
    'f1': 0.43243243243243246},
   'applications': {'recall': 0.5975609756097561,
    'precision': 0.8352272727272727,
    'f1': 0.6966824644549763}},
  'links': {'name_of_mof|||mof_formula': {'recall':np.nan,
    'precision':np.nan,
    'f1':np.nan},
   'name_of_mof|||mof_description': {'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0},
   'name_of_mof|||guest_species': {'precision': 0.6363636363636364,
    'recall': 0.20588235294117646,
    'f1': 0.3111111111111111},
   'name_of_mof|||applications': {'precision': 0.6392405063291139,
    'recall': 0.36200716845878134,
    'f1': 0.4622425629290618}},
  'support': {'ents': {'name_of_mof': 65,
    'mof_formula': 16,
    'mof_description': 22,
    'guest_species': 26,
    'applications': 128},
   'words': {'name_of_mof': 67,
    'mof_formula': 56,
    'mof_description': 77,
    'guest_species': 26,
    'applications': 246},
   'links_ents': {'name_of_mof|||mof_formula': 6,
    'name_of_mof|||mof_description': 16,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 137},
   'links_words': {'name_of_mof|||mof_formula': 10,
    'name_of_mof|||mof_description': 38,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 279}},
  'exact_match': 0.13725490196078433,
  'JWsim': 0.9213918970540684,
  'parsable': 1.0},
 '4k': {'ents': {'name_of_mof': {'recall': 0.5970149253731343,
    'precision': 0.7407407407407407,
    'f1': 0.6611570247933884},
   'mof_formula': {'recall': 0.35714285714285715,
    'precision': 0.6666666666666666,
    'f1': 0.46511627906976744},
   'mof_description': {'recall': 0.36363636363636365,
    'precision': 0.4827586206896552,
    'f1': 0.4148148148148148},
   'guest_species': {'recall': 0.3076923076923077,
    'precision': 0.6666666666666666,
    'f1': 0.42105263157894735},
   'applications': {'recall': 0.5650406504065041,
    'precision': 0.7722222222222223,
    'f1': 0.6525821596244131}},
  'links': {'name_of_mof|||mof_formula': {'recall':np.nan,
    'precision':np.nan,
    'f1':np.nan},
   'name_of_mof|||mof_description': {'precision': 0.5714285714285714,
    'recall': 0.10526315789473684,
    'f1': 0.17777777777777778},
   'name_of_mof|||guest_species': {'precision': 0.4375,
    'recall': 0.20588235294117646,
    'f1': 0.28},
   'name_of_mof|||applications': {'precision': 0.5194805194805194,
    'recall': 0.2867383512544803,
    'f1': 0.36951501154734406}},
  'support': {'ents': {'name_of_mof': 65,
    'mof_formula': 16,
    'mof_description': 22,
    'guest_species': 26,
    'applications': 128},
   'words': {'name_of_mof': 67,
    'mof_formula': 56,
    'mof_description': 77,
    'guest_species': 26,
    'applications': 246},
   'links_ents': {'name_of_mof|||mof_formula': 6,
    'name_of_mof|||mof_description': 16,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 137},
   'links_words': {'name_of_mof|||mof_formula': 10,
    'name_of_mof|||mof_description': 38,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 279}},
  'exact_match': 0.17647058823529413,
  'JWsim': 0.9261196516334903,
  'parsable': 0.9803921568627451},
 '8k': {'ents': {'name_of_mof': {'recall': 0.5522388059701493,
    'precision': 0.8043478260869565,
    'f1': 0.6548672566371682},
   'mof_formula': {'recall': 0.2857142857142857,
    'precision': 0.8888888888888888,
    'f1': 0.43243243243243246},
   'mof_description': {'recall': 0.36363636363636365,
    'precision': 0.3835616438356164,
    'f1': 0.37333333333333335},
   'guest_species': {'recall': 0.3076923076923077,
    'precision': 0.7272727272727273,
    'f1': 0.43243243243243246},
   'applications': {'recall': 0.5650406504065041,
    'precision': 0.7595628415300546,
    'f1': 0.6480186480186481}},
  'links': {'name_of_mof|||mof_formula': {'recall':np.nan,
    'precision':np.nan,
    'f1':np.nan},
   'name_of_mof|||mof_description': {'precision': 0.4,
    'recall': 0.10526315789473684,
    'f1': 0.16666666666666666},
   'name_of_mof|||guest_species': {'precision': 0.6363636363636364,
    'recall': 0.20588235294117646,
    'f1': 0.3111111111111111},
   'name_of_mof|||applications': {'precision': 0.564935064935065,
    'recall': 0.3118279569892473,
    'f1': 0.40184757505773666}},
  'support': {'ents': {'name_of_mof': 65,
    'mof_formula': 16,
    'mof_description': 22,
    'guest_species': 26,
    'applications': 128},
   'words': {'name_of_mof': 67,
    'mof_formula': 56,
    'mof_description': 77,
    'guest_species': 26,
    'applications': 246},
   'links_ents': {'name_of_mof|||mof_formula': 6,
    'name_of_mof|||mof_description': 16,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 137},
   'links_words': {'name_of_mof|||mof_formula': 10,
    'name_of_mof|||mof_description': 38,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 279}},
  'exact_match': 0.13725490196078433,
  'JWsim': 0.9159589615436909,
  'parsable': 0.9411764705882353},
 '13812': {'ents': {'name_of_mof': {'recall': 0.6567164179104478,
    'precision': 0.7719298245614035,
    'f1': 0.7096774193548387},
   'mof_formula': {'recall': 0.30357142857142855,
    'precision': 0.8095238095238095,
    'f1': 0.44155844155844154},
   'mof_description': {'recall': 0.2727272727272727,
    'precision': 0.4883720930232558,
    'f1': 0.35},
   'guest_species': {'recall': 0.5769230769230769,
    'precision': 0.7894736842105263,
    'f1': 0.6666666666666666},
   'applications': {'recall': 0.6016260162601627,
    'precision': 0.8131868131868132,
    'f1': 0.6915887850467289}},
  'links': {'name_of_mof|||mof_formula': {'recall':np.nan,
    'precision':np.nan,
    'f1':np.nan},
   'name_of_mof|||mof_description': {'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0},
   'name_of_mof|||guest_species': {'precision': 0.6086956521739131,
    'recall': 0.4117647058823529,
    'f1': 0.4912280701754386},
   'name_of_mof|||applications': {'precision': 0.6951219512195121,
    'recall': 0.40860215053763443,
    'f1': 0.5146726862302482}},
  'support': {'ents': {'name_of_mof': 65,
    'mof_formula': 16,
    'mof_description': 22,
    'guest_species': 26,
    'applications': 128},
   'words': {'name_of_mof': 67,
    'mof_formula': 56,
    'mof_description': 77,
    'guest_species': 26,
    'applications': 246},
   'links_ents': {'name_of_mof|||mof_formula': 6,
    'name_of_mof|||mof_description': 16,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 137},
   'links_words': {'name_of_mof|||mof_formula': 10,
    'name_of_mof|||mof_description': 38,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 279}},
  'exact_match': 0.1568627450980392,
  'JWsim': 0.9167839948837687,
  'parsable': 0.9803921568627451},
 'llama2': {'ents': {'name_of_mof': {'recall': 0.6865671641791045,
    'precision': 0.7796610169491526,
    'f1': 0.7301587301587301},
   'mof_formula': {'recall': 0.5178571428571429,
    'precision': 0.7073170731707317,
    'f1': 0.5979381443298969},
   'mof_description': {'recall': 0.4805194805194805,
    'precision': 0.5138888888888888,
    'f1': 0.4966442953020134},
   'guest_species': {'recall': 0.6923076923076923,
    'precision': 0.9,
    'f1': 0.782608695652174},
   'applications': {'recall': 0.6422764227642277,
    'precision': 0.6422764227642277,
    'f1': 0.6422764227642277}},
  'links': {'name_of_mof|||mof_formula': {'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0},
   'name_of_mof|||mof_description': {'precision': 0.5,
    'recall': 0.6052631578947368,
    'f1': 0.5476190476190477},
   'name_of_mof|||guest_species': {'precision': 1.0,
    'recall': 0.5,
    'f1': 0.6666666666666666},
   'name_of_mof|||applications': {'precision': 0.42402826855123676,
    'recall': 0.43010752688172044,
    'f1': 0.4270462633451958}},
  'support': {'ents': {'name_of_mof': 65,
    'mof_formula': 16,
    'mof_description': 22,
    'guest_species': 26,
    'applications': 128},
   'words': {'name_of_mof': 67,
    'mof_formula': 56,
    'mof_description': 77,
    'guest_species': 26,
    'applications': 246},
   'links_ents': {'name_of_mof|||mof_formula': 6,
    'name_of_mof|||mof_description': 16,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 137},
   'links_words': {'name_of_mof|||mof_formula': 10,
    'name_of_mof|||mof_description': 38,
    'name_of_mof|||guest_species': 34,
    'name_of_mof|||applications': 279}},
  'exact_match': 0.09803921568627451,
  'JWsim': 0.9359487949258544,
  'parsable': 1.0}}
# %%
mapname = {'llamat3_chat_mof2_4k.pkl':'4k', 'llamat3_chat_mof2_8k.pkl':'8k', 'llamat3_chat_mof2_13k.pkl': '13k', 'llamat3_chat_mof2_13812.pkl':'13812',  'mof2_test_llama2.pkl':'llama2'}
final_results = {}; #//DO not uncomment.
# curname = mapname[current];
# curname = '_'.join(current.split('_')[1:3]);
curname = sys.argv[1];
# if('orca' in current):
#     curname = '_'.join(current.split('_')[1:5]);
final_results[curname] = r_dict_avg; 
final_results[curname]['support'] = support; ##Adding support to the dictionary.
final_results[curname]["exact_match"] = np.mean(all_exact_match_accuracy)
final_results[curname]["JWsim"] = np.mean(all_winkler_similarities)
final_results[curname]["parsable"] = 1-np.mean(all_unparsable)

# %%


# %%
final_results.keys()

# %%
type(0.234) == float

# %%
num = 0;
def printR(x="", end = "\n"):
    if(type(x) == float):
        print(round(x,3), end = end);
    else:
        print(x, end = end);
others = ["exact_match","JWsim", "parsable"]
for model in final_results.keys():
    if(num == 0): ### Prints the column names.
        printR("model,",end="");
        for tasks in final_results[model]['ents'].keys():
            printR(tasks, end = ",");
        for task in others:
            printR(task, end =",");
        print(); 
    num += 1;
    printR(model, end = ",");
    for task in final_results[model]['ents'].keys():
        printR(round(final_results[model]['ents'][task]['f1'],3), end=",");
    for task in others:
        printR(round(final_results[model][task],3), end = ",");
    print();
print(*["*" for _ in range(20)]);
print("entity-links:");
# %%
consider = ['name_of_mof|||mof_formula',	'name_of_mof|||guest_species',	'name_of_mof|||applications', 'name_of_mof|||mof_description'];
# consider = ['formula|||applications', 'formula|||description', 'formula|||structure_or_phase']


print(*["*" for _ in range(20)]);
print("entity-links:");
# %%
#consider = ['formula|||applications', 'formula|||description', 'formula|||structure_or_phase']
#consider = ['formula|||acronym', 'formula|||applications', 'formula|||description', 'formula|||name', 'formula|||structure_or_phase']


k = list(final_results.keys())[0];
for j in consider:
    print(j, end = ", ");
    print(round(final_results[k]['links'][j]['precision'], 3), end = ", ");
    print(round(final_results[k]['links'][j]['recall'], 3), end = ", ");
    print(round(final_results[k]['links'][j]['f1'], 3), end = ", ");
    print();

print("*"*20);
cnt = 0; 
for k in final_results.keys():
    if cnt == 0:
        print("model, ", end = "");
        for j in consider:
            print(j, end = ", ");
        # print("parsable");
        print();
    cnt += 1;
    print(k, ",", end ="");
    # for j in results_[k]['links'].keys():
    # 
    for j in consider:
        print(round(final_results[k]['links'][j]['f1'], 3), end = ", ");
    # print(f"{results_[k]['parsable'][0]}/{results_[k]['parsable'][1]}");
    print();    
