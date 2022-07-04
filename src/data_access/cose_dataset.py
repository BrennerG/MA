import json
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
from data_access.locations import LOC
from functools import reduce
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU


class CoseDataset(Dataset):

    def __init__(self, mode='train'):
        self.parselabel = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        self.mode = mode
        self.location = 'cose'
        # data
        self.train, self.val, self.test = EU.load_datasets(LOC['cose'])
        if mode == 'train': self.annotations = self.train
        elif mode == 'val': self.annotations = self.val
        elif mode == 'test': self.annotations = self.test
        # docs
        self.docids = [x.annotation_id for x in self.annotations]
        self.docs = EU.load_flattened_documents(LOC['cose'], self.docids)
        self.labels = [self.parselabel[x.classification] for x in self.annotations]

        self.data = []
        for X in self.annotations:
            self.data.append((self.docs[X.annotation_id], X.query_type, X.query.split(' [sep] '), float(self.parselabel[X.classification]), list(X.evidences)[0][0]))

        self.avg_evidence_len = round(np.mean([len(x[4].text.split()) for x in self]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    # modes = {'comprehensiveness', 'sufficiency'}
    def erase(self, weights:[], k=None, mode='comprehensiveness'):
        assert len(weights) == len(self.data)
        if k == None: k = round(np.mean([len(x[4].text.split()) for x in self]))
        weights = [tensor.detach().numpy() for tensor in weights] # TODO make models return detached vector?

        erased = []
        for i, X in enumerate(self): # (Question, Context, Answer, Label, Evidence)
            assert len(weights[i]) >= len(X[0]) # for debugging: if not asserts: FIX IT

            # get indices with highest weights
            if len(weights[i]) >= k: idx_k = np.argpartition(weights[i], -k)[-k:]
            else: idx_k = range(len(weights[i]))

            # erase tokens depending on mode
            if mode == 'comprehensiveness':
                erased.append( ([x for (i,x) in enumerate(X[0]) if i not in idx_k], X[1], X[2], X[3], X[4]) )
            elif mode == 'sufficiency':
                erased.append( ([x for (i,x) in enumerate(X[0]) if i in idx_k], X[1], X[2], X[3], X[4]) )
            else:
                raise AttributeError('mode unknown!')

        erased_dataset = CoseDataset(mode=self.mode)
        erased_dataset.data = erased
        return erased_dataset