import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GAT, GATConv
from models.albert_embedding import AlbertEmbedding


class BERT_GAT(torch.nn.Module):

    def __init__(self, params:{}):
        super().__init__()
        self.embedding = AlbertEmbedding(params)
        self.num_heads = params['num_heads'] if 'num_heads' in params else None
        self.gconv0 = GATConv(in_channels=768, out_channels=300, heads=self.num_heads, dropout=params['dropout'])
        self.relu0 = nn.ReLU()
        self.gconv1 = GATConv(in_channels=600, out_channels=64, heads=self.num_heads, dropout=params['dropout'])
        self.relu1 = nn.ReLU()
        self.gconv2 = GATConv(in_channels=128, out_channels=5, heads=1, dropout=params['dropout'])
        self.relu2 = nn.ReLU()
    
    def forward(self, data, **args):
        for i,ans in enumerate(data['answers']):
            tokens = data['concept_ids'][i]
            emb, bert_map = self.embedding(" ".join(tokens), return_bert_map=True)
        return None
    
    def __call__(self, data, **args):
        if isinstance(data, dict): # single sample
            return self.forward(data, **args)
        elif isinstance(data, list): # data set
            outputs = []
            for sample in data:
                outputs.append(self.forward(sample, **args))
            return outputs