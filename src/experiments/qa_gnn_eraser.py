import sys
import csv
import pickle
import numpy as np
import json
import os
import torch

from tqdm import tqdm
from datasets import load_dataset
from copy import copy

from data.locations import LOC
import evaluation.eval_util as E


CPNET_VOC_PATH = "data/qa_gnn/concept.txt"

TRAIN_PREDS_PATH = 'data/qa_gnn/NEW_train_preds_20221027145654.csv'
DEV_PREDS_PATH = 'data/qa_gnn/NEW_dev_preds_20221027151911.csv'
TEST_PREDS_PATH = 'data/qa_gnn/NEW_test_preds_20221027151244.csv'

TRAIN_SET_PATH = 'data/qa_gnn/train_set.pickle'
DEV_SET_PATH = 'data/qa_gnn/dev_set.pickle'
TEST_SET_PATH = 'data/qa_gnn/test_set.pickle'

GROUNDED_TRAIN_PATH = 'data/qa_gnn/train.grounded.jsonl'
GROUNDED_DEV_PATH = 'data/qa_gnn/dev.grounded.jsonl'
GROUNDED_TEST_PATH = 'data/qa_gnn/test.grounded.jsonl'

STATEMENTS_TRAIN_PATH = 'data/qa_gnn/train.statement.jsonl'
STATEMENTS_DEV_PATH = 'data/qa_gnn/dev.statement.jsonl'
STATEMENTS_TEST_PATH = 'data/qa_gnn/test.statement.jsonl'

LINKED_DATA_PATH = "data/qa_gnn/linked_data_{}.json"

ERASED_DATA_LOC = "data/qa_gnn/erased"


def get_preds(PATH, BATCH_SIZE):
    preds = {}
    csv.field_size_limit(sys.maxsize) # increase csv max reading size
    with open(PATH) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        for row in tqdm(spamreader, desc='reading predictions file'):
            qids = [x.strip() for x in row[0].replace("'","").replace('[','').replace(']','').split(',')]
            assert len(qids) <= BATCH_SIZE, 'ERROR: incorrect batch size'
            logits_str = row[1].replace(']','').replace('[','').split(',')
            logits = np.array([float(x) for x in logits_str]).reshape(-1,5)
            attn_str = row[2].replace(']','').replace('[','').split(',')
            attn = np.array([float(x) for x in attn_str]).reshape(-1,2,5,200)
            for b in range(len(qids)):
                preds[qids[b]] = (logits[b], attn[b])
    return preds

# run ERASER
def run(DATASET_CHOICE, TIMESTAMP):
    if DATASET_CHOICE == 'train':
        preds_path = TRAIN_PREDS_PATH
        comp_preds_path = 'data/qa_gnn/erased_predictions/comp/COMP_train_preds_20221031180500.csv'
        suff_preds_path = 'data/qa_gnn/erased_predictions/suff/SUFF_train_preds_20221103104520.csv'
        BATCH_SIZE = 32
    elif DATASET_CHOICE == 'dev':
        preds_path = DEV_PREDS_PATH
        comp_preds_path = 'data/qa_gnn/erased_predictions/comp/COMP_dev_preds_20221031181720.csv'
        suff_preds_path = 'data/qa_gnn/erased_predictions/suff/SUFF_dev_preds_20221103104013.csv'
        BATCH_SIZE = 1
    elif DATASET_CHOICE == 'test':
        preds_path = TEST_PREDS_PATH
        comp_preds_path = 'data/qa_gnn/erased_predictions/comp/COMP_test_preds_20221101154716.csv'
        suff_preds_path = 'data/qa_gnn/erased_predictions/suff/SUFF_test_preds_20221103105610.csv'
        BATCH_SIZE = 1

    # INIT: GET ALL OF THESE
    raw_pred = get_preds(preds_path, BATCH_SIZE)
    raw_comp_pred = get_preds(comp_preds_path, BATCH_SIZE)
    raw_suff_pred = get_preds(suff_preds_path, BATCH_SIZE)

    assert raw_comp_pred.keys() == raw_suff_pred.keys()
    assert len(raw_pred) >= len(raw_comp_pred)

    doc_ids = list(raw_comp_pred.keys())
    choices = [np.argmax(raw_pred[qid][0],axis=0) for qid in doc_ids]
    raw_attn = [raw_pred[qid][1] for qid in doc_ids]
    attn = [] # TODO could write a method, bc this is also repeated in the link_data method
    for c,a in zip(choices,raw_attn):
        v = np.mean(a[:,c],axis=0)
        attn.append(torch.Tensor(v))
    
    # normalize
    def normalize(x):
        return (x-x.min()) / (x.max()-x.min())

    # RE-SYNC comp-, suff- and regular-preds (indices)
    comp_pred = [torch.Tensor(normalize(x[0])) for x in raw_comp_pred.values()]
    suff_pred = [torch.Tensor(normalize(x[0])) for x in raw_suff_pred.values()]
    pred = [normalize(x[1][0]) for x in raw_pred.items() if x[0] in raw_comp_pred.keys()] # get rid of the qids we lost during comp-/suff-ing previously (41,1,1) for (train,dev,test)
    
    er_results = E.create_results(
        docids=doc_ids,
        predictions=pred,
        comp_predicitons=comp_pred,
        suff_predictions=suff_pred,
        attentions=attn,
        aopc_thresholded_scores=None)

    scores = E.classification_scores(results=er_results, mode='custom', with_ids=doc_ids)

    with open(f"data/qa_gnn/evaluation_{TIMESTAMP}.json", 'w') as f:
        json.dump(scores,f)

    return scores

    # er_results = E.create_results(
    #   doc_ids === type list
    #   pred === list of torch.Tensor, shape=torch.Size([5])
    #   comp_pred === same as pred
    #   suff_pred === same as pred
    #   attn ===  list of torch.Tensor
    #   aopc_thresholded_scores=aopc_predictions) === None

    # return E.classification_scores(
    #   results=er_results, === above
    #   mode='val',  === create mode='custom'
    #   aopc_thresholds=self.params['aopc_thresholds'], === not necessary?
    #   with_ids=doc_ids) === see above

def erase(DATASET_CHOICE, DIRECTORY_STAMP):
    # load og
    og_dataset = load_dataset(LOC['cose_huggingface'])
    # load linked data
    with open(LINKED_DATA_PATH.format(DATASET_CHOICE)) as f:
        data = json.load(f)
    # make directory
    DIR_PATH = f"{ERASED_DATA_LOC}/{DIRECTORY_STAMP}"
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    # erase tokens 
    comp_file = open(f"{DIR_PATH}/{DATASET_CHOICE}_comp.jsonl", "a")
    suff_file = open(f"{DIR_PATH}/{DATASET_CHOICE}_suff.jsonl", "a")
    for qid,lsample in tqdm(data.items()):
        logits = np.array(lsample['logits'])
        attn = np.array(lsample['attention'])
        ogsample = get_from(qid, og_dataset)
        tokens = ogsample['question'].split()
        assert len(attn) == len(tokens)
        indices = attn.nonzero()[0]

        # comprehensiveness
        comp_tokens = tokens.copy()
        for index in reversed(indices):
            del comp_tokens[index]

        # sufficiency
        suff_tokens = []
        for index in indices:
            suff_tokens.append(tokens[index])

        # form dataset
        res_sample = {
	        "id": qid,
            "question": {
                "question_concept": ogsample['context'],
                "choices": [
                    {"label": "A", "text": ogsample['answers'][0]},
                    {"label": "B", "text": ogsample['answers'][1]},
                    {"label": "C", "text": ogsample['answers'][2]},
                    {"label": "D", "text": ogsample['answers'][3]},
                    {"label": "E", "text": ogsample['answers'][4]}],
                "stem": None
            }
        }

        if DATASET_CHOICE != 'test':
            res_sample["answerKey"] = ["A","B","C","D","E"][ogsample['label']]

        # save datasets
        res_sample['question']['stem'] = ' '.join(comp_tokens)
        print (json.dumps(res_sample), file=comp_file)
        res_sample['question']['stem'] = ' '.join(suff_tokens)
        print (json.dumps(res_sample), file=suff_file)

    return None

def get_from(qid, dataset, ds_type='og'):
    assert ds_type == 'og'
    for split in dataset.keys():
        for sample in dataset[split]:
            if sample['id'] == qid:
                return sample
    raise LookupError(f"ERROR: sample {qid} not in {ds_type} dataset")

def link_dataset(DATASET_CHOICE='dev'):

    if DATASET_CHOICE == 'train':
        QAGNN_SET_PATH = TRAIN_SET_PATH
        QAGNN_PREDS_PATH = TRAIN_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_TRAIN_PATH
        QAGNN_STATEMENT_PATH = STATEMENTS_TRAIN_PATH
        BATCH_SIZE = 32

    elif DATASET_CHOICE == 'dev':
        QAGNN_SET_PATH = DEV_SET_PATH
        QAGNN_PREDS_PATH = DEV_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_DEV_PATH
        QAGNN_STATEMENT_PATH = STATEMENTS_DEV_PATH
        BATCH_SIZE = 1

    elif DATASET_CHOICE == 'test':
        QAGNN_SET_PATH = TEST_SET_PATH
        QAGNN_PREDS_PATH = TEST_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_TEST_PATH
        QAGNN_STATEMENT_PATH = STATEMENTS_TEST_PATH
        BATCH_SIZE = 1

    # GET OG DATASET
    og_dataset = load_dataset(LOC['cose_huggingface'])
    
    # GET CONCEPTNET VOCABULARY
    with open(CPNET_VOC_PATH) as f:
        cpnet_voc = [line.rstrip() for line in f]

    # GET QAGNN DATASET
    with open(QAGNN_SET_PATH, 'rb') as handle:
        dataset = pickle.load(handle)
    
    # GET PREDS
    # TODO substitute and test this with get_preds(..)
    csv.field_size_limit(sys.maxsize) # increase csv max reading size
    preds = {}
    with open(QAGNN_PREDS_PATH) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        for row in tqdm(spamreader, desc='reading predictions file'):
            qids = [x.strip() for x in row[0].replace("'","").replace('[','').replace(']','').split(',')]
            assert len(qids) <= BATCH_SIZE, 'ERROR: incorrect batch size'
            logits_str = row[1].replace(']','').replace('[','').split(',')
            logits = np.array([float(x) for x in logits_str]).reshape(-1,5)
            attn_str = row[2].replace(']','').replace('[','').split(',')
            attn = np.array([float(x) for x in attn_str]).reshape(-1,2,5,200)
            for b in range(len(qids)):
                preds[qids[b]] = (logits[b], attn[b])

    # EXTRACT CONCEPT STRINGS FROM DATASET
    concepts = {}
    for qids, labels, *inputs in dataset:
        # from og qa_gnn
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        #edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        #adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]
        # get concepts from prediction
        for qid, batch_node_type_ids, batch_concept_ids in zip(qids, node_type_ids.view(-1,5,200), concept_ids.view(-1,5,200)):
            pred_choice = np.argmax(preds[qid][0])
            concept_strings = [cpnet_voc[x] for x in batch_concept_ids[pred_choice]]
            concepts[qid] = (concept_strings, batch_node_type_ids[pred_choice])
 
    assert concepts.keys() == preds.keys()
    
    # GET STATEMENTS FOR GROUNDED IDS!
    grounded_ids = []
    for file in [STATEMENTS_TRAIN_PATH, STATEMENTS_DEV_PATH, STATEMENTS_TEST_PATH]:
        with open(file) as f:
            for line in f:
                grounded_ids.append(json.loads(line)['id'])
 
    # GET GROUNDED QAs
    raw_grounded = []
    for file in [GROUNDED_TRAIN_PATH, GROUNDED_DEV_PATH, GROUNDED_TEST_PATH]:
        with open(file) as f:
            for line in f:
                raw_grounded.append(json.loads(line))
    # reshape grounded
    grounded = list(np.array(raw_grounded).reshape(-1,5))
    grounded_dict = dict(zip(grounded_ids, grounded))
    assert len([x for x in preds.keys() if x in grounded_ids]) == len(preds.keys())
    # grounded_ids ist 9741, but every id from preds is in grounded_ids and vice versa...
    
    # LINK (og_dataset, qagnn_dataset, grounded)
    # TODO we could also check for n_grams, since attn weights are concerned with n-grams of tokens
    linked_data = {}
    e=0
    w=0
    for qid in tqdm(concepts.keys(), desc='linking:'):
        # init
        try:
            og_sample = get_from(qid, og_dataset)
            grounded_sample = grounded_dict[qid]
        except (LookupError, KeyError) as error:
            print(f"WARNING: could not find sample {qid}. omitting...")
            e+=1
            continue
        concept_tokens, concept_types = concepts[qid] # only of prediction
        logits, attention = preds[qid]
        pred_choice = np.argmax(logits)
        if concept_types.tolist().count(0) != len(grounded_sample[pred_choice]['qc']):
            print(f"WARNING: number of concept types doesnt match concepts @ {qid}")
            w+=1
        # aggregate attention
        # TODO use aggregate_attention(..)
        pred_attention = attention[:,pred_choice]
        agg_attention = np.mean(pred_attention, axis=0) # TODO if there's more experimentation on original arch, use the actual method for this!
        # link tokens with attention scores
        relevant_indices = (concept_types==0).nonzero().squeeze().tolist()
        relevant_attention = agg_attention.take(relevant_indices) # zip with tokens
        if isinstance(relevant_attention, np.float64): relevant_attention = [relevant_attention]
        token_attn_dict = dict(zip(grounded_sample[pred_choice]['qc'], relevant_attention))
        linked_attention = [token_attn_dict[x] if x in token_attn_dict else 0.0 for x in og_sample['question'].split()]
        linked_data[qid] = {
            'logits': logits.tolist(),
            'attention': linked_attention
        }

    # SAVE
    with open(f"data/qa_gnn/linked_data_{DATASET_CHOICE}.json", 'w', encoding ='utf8') as json_file:
        json.dump(linked_data, json_file, allow_nan=False)

    return linked_data