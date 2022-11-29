import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

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
        self.bert_dim = 768
        self.dropout_rate = 0.2

        # MODULES
        self.activation = nn.GELU()
        self.embedding = AlbertEmbedding(params)
        self.sentence_linear = nn.Linear(self.bert_dim, self.gat_hidden_dim)
        self.concept_emb = None # nn.Embedding(None, 1024)
        self.concept_linear = nn.Linear(1024, self.gat_hidden_dim)
        self.pre_gnn_dropout = nn.Dropout(self.dropout_rate)
        self.gat_layers = GatModule(params) # TODO
        self.pooler = Pooler(params,self.bert_dim,self.gat_hidden_dim)
        self.fc_dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.gat_hidden_dim*2 + self.bert_dim,1)
   

    def forward(self, data, **args):
        assert isinstance(data,dict) # single sample needed, iterable_forward not implemented!
        softmax_logits = args['softmax_logits'] if 'softmax_logits' in args else True
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
            edge_index = self.relative_edges(concept_tokens, data['edges'][i])
            edge_index = torch.tensor(edge_index).T.to(self.device)
            gnn_input = self.pre_gnn_dropout(torch.cat((sentence_node,nodes),dim=0))
            gnn_output, edge_attn = self.gat_layers(gnn_input, edge_index)

            # POOLING
            pooled_output, node_attn = self.pooler(sent_vec, gnn_output)  # TODO node_attn = [n_head, n_nodes]
            agg_attn = self.aggregate_node_attention(data['nodes_to_qa_tokens'][i], node_attn, qa_tokens)
            attentions.append(agg_attn)

            # FC
            z = gnn_output[0]
            concat = self.fc_dropout(torch.cat((pooled_output.squeeze(), sent_vec.squeeze(), z), 0))
            proba_vec[i] = self.fc(concat)

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
    
    def relative_edges(self, concept_tokens, edge_index):
        rel_edge_index = []
        for a,b in edge_index:
            _a = concept_tokens.index(self.GP.id2concept[a])
            _b = concept_tokens.index(self.GP.id2concept[b])
            rel_edge_index.append([_a,_b])
        return rel_edge_index
    
    def aggregate_node_attention(self, nodes_to_qa_tokens, node_attn, qa_tokens):
        agg_node_attn = node_attn.mean(dim=0)
        A = np.zeros(len(qa_tokens))
        for bert_pos,token_pos in enumerate(nodes_to_qa_tokens):
            if token_pos != None and token_pos != len(A):
                A[token_pos] = agg_node_attn[bert_pos]
        return A
            
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

    def __init__(self, params, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        n_head = params['num_heads'] if 'num_heads' in params else None
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin

        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head
        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, q, k): # sent_vec, gnn_output
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        # if mask is not None: mask = mask.repeat(n_head, 1)

        output, attn = self.attention(qs, ks, vs, mask=None)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn