import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from locations import LOC
from functools import reduce

class CoseDataset(Dataset):

    def __init__(self, path_to_raw:str=LOC['cose_train'], path_to_docs:str=LOC['cose_docs'], limit=-1):
        self.location = path_to_raw
        self.data = []
        self.docs = {}

        # LOAD MAIN DATA
        with open(path_to_raw, 'r') as json_file:
            json_list = list(json_file)

            #for json_str in json_list:
            if limit > 0:
                for json_str in json_list[:limit]:
                    result = json.loads(json_str)
                    self.data.append(result)
            else:
                for json_str in json_list:
                    result = json.loads(json_str)
                    self.data.append(result)
            
        # LOAD DOCS (QUESTIONS)
        with open(path_to_docs, 'r') as json_file:
            json_list = list(json_file)

            for json_str in json_list:
                result = json.loads(json_str)
                self.docs[result['docid']] = result['document']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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


if __name__ == "__main__":
    ds = CoseDataset(LOC['cose_train'])
    ts = CoseDataset(LOC['cose_test'])
    x = ds[0]
    print('break')