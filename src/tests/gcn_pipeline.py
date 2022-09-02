import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.utils.random  import barabasi_albert_graph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

'''
TODO
~~1 run the most simple classifcation GNN with dummy data~~
2 connect real data
    - convert text to graph (random embedding)
    - attach non-BERT embedding
3 classify real data with GCN (only see that it runs, results will be poop)
4 Message Passing
'''


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.mean(dim=0) # same as torch_geometric.nn.pool.glob.global_mean_pool
        return F.softmax(x)

def run():
    # GENERATING RANDOM DUMMY DATA
    num_samples = 11
    num_nodes = 30
    num_edges = 3
    emb_dim =  300
    num_classes = 5
    data = [None] * num_samples
    ys = torch.randint(0, num_classes, (num_samples,))
    for i in range(num_samples):
        edge_index = barabasi_albert_graph(num_nodes, num_edges)
        x = torch.rand(num_nodes, 300) # dummy embeddings
        y = ys[i]
        data[i] = Data(x=x, y=y, edge_index=edge_index)

    # TRAIN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(emb_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = CrossEntropyLoss()
    model.train()

    # LOOP
    for epoch in range(10):
        preds = torch.Tensor(num_samples)
        for i,sample in enumerate(data):
            sample = sample.to(device)
            optimizer.zero_grad()
            out = model(sample)
            preds[i] = torch.argmax(out)
            loss = loss_fn(out, sample.y)
            loss.backward()
            optimizer.step()
    
    # EVALUATE
    model.eval()
    acc = Accuracy(num_classes=num_classes)
    print(acc(preds.int(), ys.int()))
    print('done')