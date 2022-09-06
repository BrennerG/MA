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
from datasets import load_dataset
from data.locations import LOC
from tqdm import tqdm

from tests.ud_preproc import UDParser
from tests.glove import GloveEmbedder
 

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, device='cuda:0'):
        super().__init__()
        self.device = device
        self.num_node_features = num_node_features
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16,1)
        self.embedding= GloveEmbedder(LOC['glove_embedding'])

    def forward(self, data):
        proba_vec = torch.zeros(5)
        for i,answer in enumerate(data[0]['answers']):
            qa = f"{data[0]['question']} {answer}"
            emb = self.embedding(qa).to(self.device)
            edge_index = torch.Tensor(data[1][i]).T.long().to(self.device)
            x = self.conv1(emb, edge_index) 
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            proba_vec[i] = x.mean(dim=0) # same as torch_geometric.nn.pool.glob.global_mean_pool
        return F.softmax(proba_vec, dim=0)

def run():
    # GETTING THE DATA
    num_samples= -1
    emb_dim =  300
    split='train'
    cose = load_dataset(LOC['cose_huggingface'])
    dataset = cose[split]
    udparser = UDParser()
    edges = udparser(dataset, num_samples=num_samples, split=split)
    data = list(zip(list(dataset),edges))

    # TRAIN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(emb_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = CrossEntropyLoss()
    model.train()

    # LOOP
    for epoch in range(10):
        preds = torch.zeros(len(data))
        for i,sample in enumerate(tqdm(data, desc=f'epoch={epoch} training...')):
            optimizer.zero_grad()
            out = model(sample)
            preds[i] = torch.argmax(out)
            loss = loss_fn(out,preds[i].long())
            loss.backward()
            optimizer.step()
    
    # EVALUATE
    model.eval()
    ys = torch.Tensor([x[0]['label'] for x in data]).int()
    acc = Accuracy(num_classes=5)
    print(acc(preds.int(), ys))
    print('done')