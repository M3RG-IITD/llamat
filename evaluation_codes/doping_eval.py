import os
import argparse
import json
import sys
import pandas as pd
from monty.serialization import loadfn
import pprint
import matplotlib.pyplot as plt
import jellyfish
import numpy as np

import random
import json
def load_jsonl(path):
    with open(path, 'r') as f:
        a = f.readlines()
        g = [json.loads(i) for i in a]
    return g

import ast
doping_train = load_jsonl('doping_train.jsonl')
doping_test = load_jsonl('doping_test.jsonl')

import pickle

useidx = 0;
if(len(sys.argv) < 2):
    print("Please provide the model output pickle file as an argument");
    sys.exit(0);
current = sys.argv[1] + "_doping_test.pkl"; #test_pkls[useidx]; 
curname = sys.argv[1]; 
split_token = '<|im_e'; ## FOR LLaMat-3 Generations
if('llama2' in current or 'llamat2' in current):
    split_token = '\n an';  ## FOR LLaMat-2 Generations
try:
    with open(current,'rb') as f:
        outputcs = pickle.load(f)
    f.close()
except Exception as e:
    print("File not found: ", current);
    exit(); 

print(f"model output: {current}, split token: {split_token}");
# %%
consider  = []
for i in range(len(doping_test)):#
    if doping_test[i]['answer'] == {'basemats': {}, 'dopants': {}, 'dopants2basemats': {}}:
        pass
    else:
        consider.append(i)

# %%
import json

doptest_original = json.load(open('doping_test_og.json'))

tnmap = dict()
for tnid, s in enumerate(doptest_original):
    ss = s['input'].split('. ### ')[0][:30]
    for idx, s2 in enumerate(doping_test):
        if ss in s2['question']:
            # print(tnid, idx)
            tnmap[idx] = tnid #len(s['input'].split('. ### ')[0].split(" "))
            s2['tn_init'] = len(doptest_original[tnid]['input'].split(' Extract doping information from this sentence. ###')[0].split())
            
tot = 0
redo = []
gold = []
test = []
tn = []
for i in consider:
    try: 
        # print(i, doping_test[i]['answer'])
        pred = ast.literal_eval(str(outputcs[i]).split(split_token)[0])
        test.append(ast.literal_eval(str(outputcs[i]).split(split_token)[0]))
        gold.append(doping_test[i]['answer'])
        tn.append(doping_test[i]['tn_init'])
        if doping_test[i]['answer'] == pred:
            tot += 1
    except:
        redo.append(i)

# %%
tot, len(redo), len(consider), len(gold), len(test)

# %%
gold[0]
idx = 197
print(outputcs[idx].replace("\n", "") \
                    .replace("  ", " ") \
                    .replace("{ ", "{") \
                    .replace(" }", "}") \
                    .replace(" .", ".") \
                    .replace("[ ", "[") \
                    .replace(" ]", "]") \
                    .strip())

outputcs[idx] = str({"basemats": {"b0": "yttrium vanadate"}, "dopants": {"d0": "europium"}, "dopants2basemats": {"d0": [ "b0"]}})
doping_test[idx]['answer']
redo2 = [5, 21, 37, 42, 43, 88, 91, 100, 102, 119, 156, 197, 207, 209, 224]
EVALUATE_MODIFIERS_AND_RESULTS = False
def evaluate(gold, test, tn, loud=False, lowercase=False):
    """
    Evaluate the performance of the model on the test set.

    Args:
        gold (list): list of dictionaries containing the gold standard annotations
        test (list): list of dictionaries containing the model's predictions (in same format)
        loud (bool): whether to print out the results of each sentence
        lowercase (bool=): if true, use lowerase

    Returns:
        scores_computed (dict): dictionary of scores by entity
        ent_categories ([str]): and list of entities used in the evaluation
        sequences_distances ([float]): The jaro winkler distances for each completion from raw string
        sequences_total (int): The total number of sequences evaluated for sequence accuracy
        sequences_correct (int): The total number of sequences exactly correct.
    """
    if EVALUATE_MODIFIERS_AND_RESULTS:
        ent_categories = ["basemats", "dopants", "results", "doping_modifiers"]
    else:
        ent_categories = ["basemats", "dopants"]

    scores = {
        k: {k2: 0 for k2 in ["tp", "tn", "fp", "fn"]} for k in ent_categories
    }

    scores["dopants2basemats"] = {"n_correct": 0, "test_retrieved": 0, "gold_retrieved": 0}

    sequences_correct = 0
    sequences_total = 0
    sequences_parsable = 0
    sequences_distances = []
    support = {
                "ents": {k: 0 for k in ent_categories},
               "words": {k: 0 for k in ent_categories},
               "links_words": 0,
                "links_ents": 0,
            }

    parsability_valid = True

    for i, val_entry in enumerate(gold):
        test_entry_tot = test[i] ##["doping_sentences"][j]["entity_graph_raw"]

        for k in ent_categories + ["dopants2basemats"]:
            if k not in test_entry_tot.keys():
                test_entry_tot[k] = []
                    
        test_entry = {k: test_entry_tot[k] for k in ent_categories + ["dopants2basemats"]}
        gold_entry = {k: val_entry[k] for k in ent_categories + ["dopants2basemats"]}

        if test_entry["dopants2basemats"] == []:
            test_entry["dopants2basemats"] = {}

        # sentence_text = s["sentence_text"]
        gold_completion = val_entry#s.get("completion", "")

        if lowercase:
            # Adjust the sequence-level scoring for seq2rel
            # lowercase the gold entries if we need to account for things in lowercase
            for k in ("dopants", "basemats"):
                for ent_id, ent_val in gold_entry[k].items():
                    gold_entry[k][ent_id] = ent_val.lower()
            # seq2rel needs some adjustment for this
            gold_completion = json.dumps({k: gold_entry[k] for k in ["dopants", "basemats", "dopants2basemats"]})

            # checking keys in test_entry
            
            test_completion = json.dumps({k: test_entry[k] for k in ["dopants", "basemats", "dopants2basemats"]})
            gold_completion = str(gold_completion).lower()\
                .replace("\n", "") \
                .replace("  ", " ") \
                .replace("{ ", "{") \
                .replace(" }", "}") \
                .replace(" .", ".") \
                .replace("[ ", "[") \
                .replace(" ]", "]") \
                .strip()
        else:
            try:
                test_completion = test[i]#["doping_sentences"][j][
                    #"llm_completion"]
            except KeyError:
                print(
                    "WARNING: Could not find completion key for test completion. Sequence-level results will be incorrect.")
                test_completion = " "
                parsability_valid = False

        if loud:
            print(s["sentence_text"])
            pprint.pprint(gold_entry)
            pprint.pprint(test_entry)



        # this is a proxy to find the unparsable sequences,
        # since by default the processing script will either throw error
        # for unparsable sequences or will pass them and return empty decoded entry
        if not str(test_completion)[-1] in ["}", ".", "\n"]:
            if loud:
                print("Sequence from LLM was likely not parsable.")
        else:
            sequences_parsable += 1

        for ent_type in ent_categories:

            gold_ents = gold_entry[ent_type]

            # correcting a relic of a previous annotation scheme
            if ent_type == "doping_modifiers":
                gold_ents_words = " ".join(gold_entry[ent_type]).split(" ")
            else:
                gold_ents_words = " ".join(list(gold_entry[ent_type].values())).split(" ")


            support["words"][ent_type] += len(gold_ents_words)
            support["ents"][ent_type] += 1 if isinstance(gold_ents, str) else len(gold_ents)

            test_ents_words = " ".join(list(test_entry[ent_type].values())).split(" ")

            gold_ents_words = [w for w in gold_ents_words if w]
            test_ents_words = [w for w in test_ents_words if w]

            if loud:
                print(ent_type, test_entry)

            # print(f"GOLD: {gold_ents_words}")
            # print(f"TEST: {test_ents_words}")

            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for w in gold_ents_words:
                if w in test_ents_words:
                    TP += 1
                else:
                    if loud:
                        print(f"FALSE NEGATIVE: {w}")
                    FN += 1

            for w in test_ents_words:
                if w not in gold_ents_words:
                    if loud:
                        print(f"FALSE POSITIVE: {w}")
                    FP += 1

            # TN = len(sentence_text.split(" ")) - TP - FN - FP
            TN = tn[i] - TP - FN - FP

            scores[ent_type]["tp"] += TP
            scores[ent_type]["tn"] += TN
            scores[ent_type]["fp"] += FP
            scores[ent_type]["fn"] += FN


        gold_entry["triplets"] = []
        test_entry["triplets"] = []

        # assemble triplets
        for is_test, rel_entry in enumerate((gold_entry, test_entry)):
            for did, bids in rel_entry["dopants2basemats"].items():
                for bid in bids:
                    try:
                        bmat_words = rel_entry["basemats"][bid]
                        dop_words = rel_entry["dopants"][did]
                        if not is_test:
                            support["links_ents"] += 1
                        for bmat_word in bmat_words.split(" "):
                            for dop_word in dop_words.split(" "):
                                if bmat_word and dop_word:
                                    rel_entry["triplets"].append(f"{bmat_word} {dop_word}")
                    except:
                        print("FAILED rel_entry = ", rel_entry);
                        continue; ## if failed then nothing to do, but continue;
        gold_triplets = gold_entry["triplets"]
        test_triplets = test_entry["triplets"]

        n_correct_triplets = 0
        for triplet in gold_triplets:
            if triplet in test_triplets:
                n_correct_triplets += 1

        scores["dopants2basemats"]["n_correct"] += n_correct_triplets
        scores["dopants2basemats"]["test_retrieved"] += len(test_triplets)
        scores["dopants2basemats"]["gold_retrieved"] += len(gold_triplets)

        support["links_words"] += len(gold_triplets)

        # Jaro winkler sequence accuracies
        dist = jellyfish.jaro_winkler_similarity(str(gold_completion), str(test_completion))
        sequences_distances.append(dist)
        sequences_total += 1

        if test_completion == gold_completion:
            if loud:
                print("Sequences are identical")
            sequences_correct += 1
        elif loud:
            print("Sequences differ:")
            print(test_completion)
            print(gold_completion)

        if loud:
            print("-"*50 + "\n")
    if loud:
        pprint.pprint(scores)

    scores_computed = {k: {} for k in ent_categories}

    for k in ent_categories:
        tp = scores[k]["tp"]
        tn = scores[k]["tn"]
        fp = scores[k]["fp"]
        fn = scores[k]["fn"]

        if tp + fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = tp / (tp + 0.5 * (fp + fn))

        print(f"{k}: prec={prec}, recall={recall}, f1={f1}")
        scores_computed[k]["precision"] = prec
        scores_computed[k]["recall"] = recall
        scores_computed[k]["f1"] = f1

    # Precision = Number of correct triples/Number of triples retrieved
    # Recall = Number of correct triples/Number of correct triples that exist in Gold set.
    # F-Measure = Harmonic mean of Precision and Recall.

    triplet_scores = scores["dopants2basemats"]
    if triplet_scores["test_retrieved"] == 0:
        triplet_prec = 0
    else:
        triplet_prec = triplet_scores["n_correct"]/triplet_scores["test_retrieved"]
    triplet_recall = triplet_scores["n_correct"]/triplet_scores["gold_retrieved"]

    if triplet_recall == 0 or triplet_prec == 0:
        triplet_f1 = 0
    else:
        triplet_f1 = (2 * triplet_prec * triplet_recall)/(triplet_prec + triplet_recall)
    print(f"triplets: prec={triplet_prec}, recall={triplet_recall}, f1={triplet_f1}")
    scores_computed["link triplets"] = {"precision": triplet_prec, "recall": triplet_recall, "f1": triplet_f1}

    return (
            scores_computed,
            ent_categories,
            sequences_distances,
            sequences_correct,
            sequences_parsable,
            sequences_total,
            support,
            parsability_valid
    )

# %%
(
         scores_computed,
         ent_categories,
         sequences_distances,
         sequences_correct,
         sequences_parsable,
         sequences_total,
         support,
         parsability_valid
     ) = evaluate(gold, test, tn, False)#, loud=False, lowercase=False)

# %%
scores_computed#,ent_categories,sequences_distances,sequences_correct,sequences_parsable,sequences_total,support,parsability_valid

# %%

    # FOR PLOTTING ONLY
# import seaborn as sns
if not parsability_valid:
    print("Sequence-level formats invalid. Skipping sequence-level metrics.")
ents_rows = []
for entc in ent_categories:
    ents_rows += [entc] * 3
df = pd.DataFrame(
    {
        "metric": ["precision", "recall", "f1"] * (len(ent_categories) + 1),
        "entity": ents_rows + ["link triplets"] * 3,
     }
)
scores_df = []
for i, r in df.iterrows():
    scores_df.append(scores_computed[r["entity"]][r["metric"]])
df["score"] = scores_df
print(df)
print("Total sequences was:", sequences_total)
print("Frac. Sequences parsable: ", sequences_parsable/sequences_total)
print("Avg sequence similarity: ", np.mean(sequences_distances))
print("Frac. of sequences exactly correct: ", sequences_correct/sequences_total)
print("Support was: ", pprint.pformat(support))
plot = False;
if plot:
    ax = sns.barplot(x="entity", y="score", hue="metric", data=df)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

results_ = {}
results_[curname] = {}; 
results_[curname]['df'] = df
results_[curname]['support'] = support;
results_[curname]['similarity']=np.mean(sequences_distances)
results_[curname]['parsable'] = sequences_parsable/sequences_total;
results_[curname]['exact_match'] = sequences_correct/sequences_total
results_.keys()
import pickle
print(*["*" for _ in range(20)]);
print("printing final tabular results now");
# %%
cur = 0;
def printR(x="", end = "\n"):
    if(type(x) == float):
        print(round(x,3), end = end);
    else:
        print(x, end = end);
print("model, basemats, dopants, triplets, similarity, exact_match, parsable");
for model in results_.keys():   
    print(model, end = ",");
    df = results_[model]['df'];
    for i in range(df.shape[0]):
        if(df.iloc[i,0] != 'f1'):
            continue;
        print(round(df.iloc[i,2], 3), end = ", ");
    print(round(results_[model]['similarity'],3), end = ",");
    printR(results_[model]['exact_match'], end = ",");
    print(round(results_[model]['parsable'],3), end = "\n");
    


