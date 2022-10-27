import csv
import pickle
import numpy as np

from tqdm import tqdm
from datasets import load_dataset

from data.locations import LOC


CPNET_VOC_PATH = "data/qa_gnn/concept.txt"

TRAIN_PREDS_PATH = "data/qa_gnn/NEW_train_preds_20221026093204.csv"
DEV_PREDS_PATH = "data/qa_gnn/NEW_dev_preds_20221026092536.csv"
TEST_PREDS_PATH = "data/qa_gnn/NEW_test_preds_20221026094635.csv"

TRAIN_SET_PATH = None
DEV_SET_PATH = 'data/qa_gnn/dev_set.pickle'
TEST_SET_PATH = None


def run():
    # GET DATASET
    with open(DEV_SET_PATH, 'rb') as handle:
        dataset = pickle.load(handle)
    
    # GET OG DATASET
    og_dataset = load_dataset(LOC['cose_huggingface'])

    # GET PREDS
    preds = {}
    with open(DEV_PREDS_PATH) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        for row in spamreader:
            qid = row[0]
            logits_str = row[1].replace(']','').replace('[','').split(',')
            logits = np.array([float(x) for x in logits_str])
            attn_str = row[2].replace(']','').replace('[','').split(',')
            attn = np.array([float(x) for x in attn_str]).reshape(2,5,200) # [nheads, nchoices, num_nodes]
            preds[qid] = (logits, attn)

    assert list(preds.keys()) == [x[0][0] for x in dataset], 'prediction qids and dataset qids dont match!'
    
    # GET CONCEPTNET VOCABULARY
    with open(CPNET_VOC_PATH) as f:
        cpnet_voc = [line.rstrip() for line in f]
    
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
        pred_choice = np.argmax(preds[qids[0]][0])
        concept_strings = [cpnet_voc[x] for x in concept_ids[pred_choice]]
        concepts[qids[0]] = (concept_strings, node_type_ids[pred_choice])
    
    # SEE OVERLAP IN OG DATASET
    overlap_counter = []
    for qid in tqdm(concepts.keys(), desc='overlap search:'):
        # find matching sample
        found = False
        for og_ds in og_dataset:
            for sample in og_dataset[og_ds]:
                if qid == sample['id']:
                    print('found!')
                    found = True

        assert found            

    return None