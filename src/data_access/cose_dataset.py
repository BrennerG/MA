import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_access.locations import LOC
from functools import reduce
import evaluation.eraserbenchmark.rationale_benchmark.utils as EU


class CoseDataset(Dataset):

    def __init__(self, mode='train'):
        self.parselabel = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        # data
        self.train, self.val, self.test = EU.load_datasets(LOC['cose'])
        self.data = None
        if mode == 'train': self.data = self.train
        elif mode == 'val': self.data = self.val
        elif mode == 'test': self.data = self.test
        # docs
        self.docids = [x.annotation_id for x in self.data]
        self.docs = EU.load_flattened_documents(LOC['cose'], self.docids)
        self.labels = [self.parselabel[x.classification] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        assert len(X.evidences) == 1
        # (question, context, answer, float(label), evidence)
        return self.docs[X.annotation_id], X.query_type, X.query.split(' [sep] '), float(self.parselabel[X.classification]), list(X.evidences)[0][0]