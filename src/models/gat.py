import torch
from torch_geometric.nn import GAT, GATConv
import torch.nn.functional as F

from models.glove import GloveEmbedder
from data.locations import LOC


class GATForMultipleChoice(torch.nn.Module):

    def __init__(self, params:{}):
        super().__init__()
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.embedding= GloveEmbedder(params, LOC['glove_embedding'])
        num_heads = 8 # TODO to params
        self.gatconv1 = GATConv(in_channels=self.embedding.dim, out_channels=params['gcn_hidden_dim'], heads=num_heads, dropout=0.1)
        self.gatconv2 = GATConv(in_channels=params['gcn_hidden_dim']*num_heads, out_channels=1, concat=False, heads=1, dropout=0.1)

    def forward(self, data):
        proba_vec = torch.zeros(5)
        attentions = []
        for i,answer in enumerate(data['answers']):
            if '?' in data['question']: # TODO do this in Dataset class?
                qa = f"{data['question']} {answer}"
            else:
                qa = f"{data['question']} ? {answer}" 
            if not self.embedding(qa).to(self.device).shape[0] > torch.max(torch.Tensor(data['qa_graphs'][i]).T.long().to(self.device)).item():
                breakpoint()
            emb = self.embedding(qa).to(self.device)
            edge_index = torch.Tensor(data['qa_graphs'][i]).T.long().to(self.device)
            x = self.gatconv1(emb, edge_index) 
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x, (edges, edge_weights) = self.gatconv2(x, edge_index, return_attention_weights=True)
            proba_vec[i] = x.mean(dim=0) # TODO experiment with pooling
            attentions.append(self.aggregate_attention(edges=edges, weights=edge_weights)) # TODO experiment with attn aggregation
        logits = F.log_softmax(proba_vec, dim=0)
        return logits, attentions[torch.argmax(logits).item()]
        
    def __call__(self, data):
        if isinstance(data, dict): # single sample
            return self.forward(data)
        elif isinstance(data, list): # data set
            outputs = []
            for sample in data:
                outputs.append(self.forward(sample))
            return outputs
    
    def aggregate_attention(self, edges, weights, mode='edge_additive', softmax=False):
        if mode != "edge_additive":
            NotImplementedError(f"aggregation mode '{mode}' not implemented yet!")
        attention = torch.zeros(edges.unique().max()+1)
        # aggregate by adding from neighbors
        for head in edges[1].tolist():
            attention[head] += weights[head].item()
        # return
        if softmax: return F.softmax(attention)
        else: return attention