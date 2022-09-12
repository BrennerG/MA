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
        # OPTION 1 
        ''' 
        self.model = GAT(
            in_channels=self.embedding.dim,
            hidden_channels=1,
            num_layers=1,
            out_channels=1
        )
        '''
        # OPTION 2
        # TODO option 2 is better coz more control!
        num_heads = 8 # TODO to params
        self.gatconv1 = GATConv(in_channels=self.embedding.dim, out_channels=params['gcn_hidden_dim'], heads=num_heads, dropout=0.1)
        self.gatconv2 = GATConv(in_channels=params['gcn_hidden_dim']*num_heads, out_channels=1, concat=False, heads=1, dropout=0.1)

    def forward(self, data):
        proba_vec = torch.zeros(5)
        for i,answer in enumerate(data['answers']):
            qa = f"{data['question']} {answer}"
            emb = self.embedding(qa).to(self.device)
            edge_index = torch.Tensor(data['qa_graphs'][i]).T.long().to(self.device)
            # OPTION 1
            # out, (edge_index_with_self_edges, edge_weights) = self.model(emb, edge_index, return_attention_weights=True) # why is out of shape (tokens,1)
            # OPTION 2
            x = self.gatconv1(emb, edge_index) 
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x, (edges, edge_weights) = self.gatconv2(x, edge_index, return_attention_weights=True)
            # pooling # TODO experiment
            proba_vec[i] = x.mean(dim=0) # same as torch_geometric.nn.pool.glob.global_mean_pool
            # TODO aggregate attention - how?
        return F.log_softmax(proba_vec, dim=0), None # TODO attention!
        
    def __call__(self, data): # TODO assert only for single sample use!
        return self.forward(data)