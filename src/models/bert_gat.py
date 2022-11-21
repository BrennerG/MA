import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GAT, GATConv
from models.albert_embedding import AlbertEmbedding


class BERT_GAT(torch.nn.Module):

    # TODO make all params passable
    # TODO don't hardcode dimensions
    def __init__(self, params:{}):
        super().__init__()

        # PARAMS
        self.device = 'cuda:0' if ('use_cuda' in params and params['use_cuda']) else 'cpu'
        self.num_heads = params['num_heads'] if 'num_heads' in params else None
        self.gat_hidden_dim = 200

        # MODULES
        self.activation = nn.GELU()
        self.embedding = AlbertEmbedding(params)
        self.sentence_linear = nn.Linear(768, self.gat_hidden_dim)
        self.concept_emb = None # nn.Embedding(None, 1024)
        self.concept_linear = nn.Linear(1024, self.gat_hidden_dim)
        self.pre_gnn_dropout = nn.Dropout(0.2)
        self.gat_layers = GatModule(params) # TODO
        self.pooler = Pooler(params) # TODO
        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1000,1)
   

    def forward(self, data, **args):
        assert isinstance(data,dict) # single sample needed, iterable_forward not implemented!
        softmax_logits = args['softmax_logits'] if 'softmax_logits' in args else False
        proba_vec = torch.zeros(5)
        attentions = []

        for i,ans in enumerate(data['answers']):
            # prepare data
            qa_tokens = f"{data['question']} {ans}".split() # OR qa_tokens = self.GP.tokenize(f"{data['question']} {ans}")
            concept_tokens = data['concept_ids'][i]
            concept_ids = torch.Tensor([self.GP.concept2id[x] for x in concept_tokens]).int().to(self.device)

            # get BERT representation
            bert_emb, bert_map = self.embedding(" ".join(concept_tokens), return_bert_map=True) 
            # tokens_bert_map = [concept_tokens[x] if x!=None else None for x in bert_map]
            # bert_map_with_4L_ids = [self.GP.concept2id[x] if x != None else None for x in tokens_bert_map]

            # create GNN representation
            sent_vec = bert_emb.mean(dim=1) # sentence representation # TODO in qagnn they take emb[:,0] for albert, but they pool properly for roberta and bert - what do?
            sentence_node = self.activation(self.sentence_linear(sent_vec))
            nodes = self.concept_emb(concept_ids)
            nodes = self.concept_linear(nodes)

            # GAT
            # edge_index = self.match_bert(data['edges'][i], bert_map, bert_map_with_4L_ids)
            edge_index = data['edges'][i]
            edge_index = torch.tensor(edge_index).T.to(self.device)
            gnn_input = self.pre_gnn_dropout(torch.cat((sentence_node,nodes),dim=0))
            gnn_output = self.gat_layers(gnn_input, edge_index)

            # POOLING
            self.pooler() 
            # TODO current

        if softmax_logits: return F.softmax(proba_vec, dim=0), attentions[torch.argmax(proba_vec)]
        return proba_vec, attentions[torch.argmax(proba_vec)]
    

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


class GatModule(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.num_heads = params['num_heads'] if 'num_heads' in params else None
        # TODO make gnn_hidden_dim passable!
        self.gconv0 = GATConv(in_channels=200, out_channels=200, heads=self.num_heads, dropout=params['dropout'])
        self.gconv1 = GATConv(in_channels=self.num_heads*200, out_channels=200, heads=self.num_heads, dropout=params['dropout'])
        self.gconv2 = GATConv(in_channels=self.num_heads*200, out_channels=200, heads=self.num_heads, dropout=params['dropout'])
        self.gconv3 = GATConv(in_channels=self.num_heads*200, out_channels=200, heads=self.num_heads, dropout=params['dropout'])
        self.gconv4 = GATConv(in_channels=self.num_heads*200, out_channels=200, heads=1, dropout=params['dropout'])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, X, edge_index):
        x = self.gconv0(X, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gconv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gconv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gconv3(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x, (edges, edge_attn) = self.gconv4(x, edge_index, return_attention_weights=True)
        x = self.activation(x)
        x = self.dropout(x)

        return x, edge_attn


class Pooler(torch.nn.Module):

    def __init__(self, params):
        pass

    def forward(self, sent_vec, gnn_output):
        pass