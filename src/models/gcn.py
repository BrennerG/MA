import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from models.glove import GloveEmbedder
from data.locations import LOC
 

class GCN(torch.nn.Module):
    def __init__(self, params:{}):
        super().__init__()
        assert 'input_dims' in params
        gcn_hidden_dim = params['input_dims'][0]
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.embedding= GloveEmbedder(params, LOC['glove_embedding'])
        self.conv1 = GCNConv(self.embedding.dim, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim,1)

    def forward(self, data): 
        assert isinstance(data,dict)
        proba_vec = torch.zeros(5)
        for i,answer in enumerate(data['answers']):
            qa = f"{data['question']} {answer}"
            emb = self.embedding(qa).to(self.device)
            edge_index = torch.Tensor(data['qa_graphs'][i]).T.long().to(self.device)
            x = self.conv1(emb, edge_index) 
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            proba_vec[i] = x.mean(dim=0) # same as torch_geometric.nn.pool.glob.global_mean_pool
        return F.log_softmax(proba_vec, dim=0), None 
    
    def __call__(self, sample, **args): # TODO feed args into forward as needed!
        return self.forward(sample)
