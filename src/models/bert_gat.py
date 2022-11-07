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
        self.gconv2 = GATConv(in_channels=128, out_channels=1, heads=1, dropout=params['dropout'])
        self.relu2 = nn.ReLU()
    
    def forward(self, data, **args):
        assert isinstance(data,dict) # single sample needed, iterable_forward not implemented!

        proba_vec = torch.zeros(5)
        attentions = []
        for i,ans in enumerate(data['answers']):
            qa_tokens = f"{data['question']} {ans}".split() # OR qa_tokens = self.GP.tokenize(f"{data['question']} {ans}")
            concept_tokens = data['concept_ids'][i]
            emb, bert_map = self.embedding(" ".join(concept_tokens), return_bert_map=True)

            # correct bert_map
            tokens_bert_map = [concept_tokens[x] if x!=None else None for x in bert_map]
            bert_map_with_4L_ids = [self.GP.concept2id[x] if x != None else None for x in tokens_bert_map]

            edge_index = self.match_bert(data['edges'][i], bert_map, bert_map_with_4L_ids)
            edge_index = torch.tensor(edge_index).T
            x = self.gconv0(emb.squeeze(), edge_index)
            x = self.relu0(x)
            x = self.gconv1(x, edge_index)
            x = self.relu1(x)
            x, (edges, edge_attn) = self.gconv2(x, edge_index, return_attention_weights=True)
            x = self.relu2(x)

            # pooling # TODO experiment
            proba_vec[i] = x.mean(dim=0)

            # aggregate edge attention # TODO experiment (how did qa-gnn do it?)
            node_attn = self.aggregate_bert_attention(edges, edge_attn, bert_map)
            token_attn = [.0] * len(qa_tokens)
            for idx,n in enumerate(data['nodes_to_qa_tokens'][i]):
                if n != None: 
                    token_attn[n] = node_attn[idx].item()
            attentions.append(token_attn)

            assert len(token_attn) == len(qa_tokens)

        return proba_vec, attentions
    
    def __call__(self, data, **args):
        if isinstance(data, dict): # single sample
            return self.forward(data, **args)
        elif isinstance(data, list): # data set
            outputs = []
            for sample in data:
                outputs.append(self.forward(sample, **args))
            return outputs
    
    def match_bert(self, edges:[], bert_map:[], bert_map_with_4L_ids:[]):
        edge_index = []
        for x in edges:
            src = [i for i,b in enumerate(bert_map_with_4L_ids) if b==x[0]]
            dst = [i for i,b in enumerate(bert_map_with_4L_ids) if b==x[1]]
            for s in src:
                for d in dst:
                    edge_index.append([s,d])
        return edge_index
    
    def aggregate_bert_attention(self, edges, weights, bert_map, mode='edge_additive', softmax=True):
        if mode != "edge_additive": NotImplementedError(f"aggregation mode '{mode}' not implemented yet!")
        attention = torch.zeros(max([x for x in bert_map if x != None])+1) # +1 for [ROOT]
        # aggregate by adding from neighbors
        for i,head in enumerate(edges[1].tolist()):
            attention[bert_map[head]] += weights[i].item() # use bert_map to map sub_tokens and their attention to the actual word tokens
        # return
        if softmax: return F.softmax(attention,dim=0)
        else: return attention