import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GAT, GATConv

from models.glove import GloveEmbedder
from data.locations import LOC


class GATForMultipleChoice(torch.nn.Module):

    def __init__(self, params:{}):
        super().__init__()
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.embedding= GloveEmbedder(params, LOC['glove_embedding'])
        num_heads = params['num_heads'] if 'num_heads' in params else None
        num_layers = params['num_layers'] if 'num_layers' in params else None
        '''
        # TODO calculate input dimensions from hidden_dim parameter 
        LAYER_DECAY = 0.5 # TODO only 0.25 and 0.5 go with num_heads of 8 - so this way of doing it is actually a bit supid, but at least calc of input vector is modular...
        inputs = [self.embedding.dim, params['gcn_hidden_dim']]
        for i in range(num_layers-2): 
            inputs.append(inputs[-1] / (LAYER_DECAY*num_heads))
        '''
        inputs = params['input_dims'][:num_layers] # meanwhile: manually # TODO test num_layers 2-7 like this
        # calculate output dimensions
        outputs = []
        for i,x in enumerate(inputs):
            if (i+1)<len(inputs): 
                y = inputs[i+1] / num_heads
            else: 
                y = 1
            outputs.append(y)
        # initialize layers
        layerlist = []
        inputs_outputs = list(zip(inputs, outputs))
        for i,o in inputs_outputs[:-1]:
            layerlist.append(GATConv(in_channels=int(i), out_channels=int(o), heads=num_heads, dropout=params['dropout']))
            layerlist.append(nn.ReLU())
        layerlist.append(GATConv(in_channels=int(inputs_outputs[-1][0]), out_channels=int(inputs_outputs[-1][1]), concat=False, heads=1, dropout=params['dropout']))
        self.layers = nn.ModuleList(layerlist)

    def forward(self, data):
        proba_vec = torch.zeros(5)
        attentions = []
        for i,answer in enumerate(data['answers']):
            if '?' in data['question']: # TODO do this in Dataset class?
                qa = f"{data['question']} {answer}"
            else:
                qa = f"{data['question']} ? {answer}" 
            x = self.embedding(qa).to(self.device)
            edge_index = torch.Tensor(data['qa_graphs'][i]).T.long().to(self.device)
            # pass through layers
            for layer in self.layers:
                if layer == self.layers[-1]: # if its last layer
                    x, (edges, edge_weights) = layer(x, edge_index, return_attention_weights=True)
                elif isinstance(layer, GATConv): # regular gat layer
                    x = layer(x, edge_index)
                elif isinstance(layer, nn.ReLU): # activation
                    x = layer(x)

            proba_vec[i] = x.mean(dim=0) # TODO experiment with pooling
            attentions.append(self.aggregate_attention(edges=edges, weights=edge_weights)) # TODO experiment with attn aggregation
        # logits = F.log_softmax(proba_vec, dim=0) # TODO CrossEntropyLoss expects true logits - not softmax!
        return proba_vec, attentions[torch.argmax(proba_vec).item()]
        
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