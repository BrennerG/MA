import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_access.locations import LOC
from functools import reduce

# TODO THIS CLASS MIGHT BE WHOLY UNNECESSARRY! (see eraserbenchmark.utils !)
class CoseDataset(Dataset):

    def __init__(self, ids, path_to_raw:str=LOC['cose_train'], path_to_docs:str=LOC['cose_docs'], return_original_form=False):
        self.location = path_to_raw
        self.data = []
        self.docs = {}
        self.return_original_form = return_original_form

        # LOAD MAIN DATA
        # TODO CURRENTLY, GET RATIONALES BY IDS OF CSQADATASET!
        with open(path_to_raw, 'r') as json_file:
            json_list = list(json_file)

            #for json_str in json_list:
            for json_str in json_list:
                result = json.loads(json_str)
                if result['annotation_id'] in ids:
                    self.data.append(result)
            
        # LOAD DOCS (QUESTIONS)
        with open(path_to_docs, 'r') as json_file:
            json_list = list(json_file)

            for json_str in json_list:
                result = json.loads(json_str)
                if result['docid'] in ids:
                    self.docs[result['docid']] = result['document']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.return_original_form:
            return self.data[idx]

        else:
            assert len(self.data[idx]['evidences']) == 1

            item = self.data[idx]
            identifier = item['annotation_id']
            question = self.docs[identifier]
            context = item['query_type']
            answer = item['query'].split(' [sep] ')
            classification = item['classification']
            label = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}[classification]
            evidence = item['evidences'][0][0]['text']

            return question, context, answer, float(label), evidence