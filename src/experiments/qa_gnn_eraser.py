import sys
import csv
import pickle
import numpy as np
import json

from tqdm import tqdm
from datasets import load_dataset

from data.locations import LOC


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

def get_from(qid, dataset, ds_type='og'):
    assert ds_type == 'og'
    for split in dataset:
        for sample in dataset[split]:
            if sample['id'] == qid:
                return sample
    raise LookupError(f"ERROR: sample {qid} not in {ds_type} dataset")

# def link_dataset(QA_GNN_STYLE_SPLIT='DEV')
def run():
    DATASET_CHOICE = 'train'

    if DATASET_CHOICE == 'train':
        QAGNN_SET_PATH = TRAIN_SET_PATH
        QAGNN_PREDS_PATH = TRAIN_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_TRAIN_PATH
        BATCH_SIZE = 32

    elif DATASET_CHOICE == 'dev':
        QAGNN_SET_PATH = DEV_SET_PATH
        QAGNN_PREDS_PATH = DEV_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_DEV_PATH
        BATCH_SIZE = 1

    elif DATASET_CHOICE == 'test':
        QAGNN_SET_PATH = TEST_SET_PATH
        QAGNN_PREDS_PATH = TEST_PREDS_PATH
        QAGNN_GROUNDED_PATH = GROUNDED_TEST_PATH
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
            attn = np.array([float(x) for x in attn_str]).reshape(-1,2,5,200).squeeze() # [nheads, nchoices, num_nodes]
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
    #dataset[0][0][0] == list(preds.keys())[0] == list(concepts.keys())[0]
    
    # GET GROUNDED QAs
    # group by num_choice
    # TODO CAREFUL ORDER OF GROUNDEDS DOESNT MATCH ANYMORE WITH concepts OR dataset...
    grounded = []
    with open(QAGNN_GROUNDED_PATH) as f:
        for qid in concepts.keys():
            group = []
            for i in range(5):
                group.append(json.loads(f.__next__()))
            grounded.append(group)
    
    # LINK (og_dataset, qagnn_dataset, grounded)
    linked_data = {}
    for i,qid in enumerate(tqdm(concepts.keys(), desc='linking:')):
        # init
        try:
            og_sample = get_from(qid, og_dataset)
        except LookupError:
            print(f"WARNING: could not find sample {qid}. omitting...")
            continue
        grounded_sample = grounded[i]
        concept_tokens, concept_types = concepts[qid] # only of prediction
        logits, attention = preds[qid]
        pred_choice = np.argmax(logits)
        assert concept_types.tolist().count(0) == len(grounded_sample[pred_choice]['qc'])
        # aggregate attention
        pred_attention = attention[:,pred_choice]
        agg_attention = np.mean(pred_attention, axis=0) # TODO if there's more experimentation on original arch, use the actual method for this!
        # link tokens with attention scores
        relevant_indices = (concept_types==0).nonzero().squeeze().tolist()
        relevant_attention = agg_attention.take(relevant_indices) # zip with tokens
        if isinstance(relevant_attention, np.float64): relevant_attention = [relevant_attention]
        assert len(grounded_sample[pred_choice]['qc']) == len(relevant_attention)
        token_attn_dict = dict(zip(grounded_sample[pred_choice]['qc'], relevant_attention))
        linked_attention = [token_attn_dict[x] if x in token_attn_dict else 0.0 for x in og_sample['question'].split()]
        linked_data[qid] = {
            'logits': logits.tolist(),
            'attention': linked_attention
        }

    # SAVE
    with open(f"data/qa_gnn/linked_data_{DATASET_CHOICE}.json", 'w', encoding ='utf8') as json_file:
        json.dump(linked_data, json_file, allow_nan=False)

    return None