import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomAttentionClassifier(nn.Module):

    def __init__(self, seed:int):
        super().__init__()
        self.TYPE = "RandomAttentionClassifier"
        self.seed = seed
        self.lin = nn.Linear(3, 1)
        torch.manual_seed(self.seed)

    def forward(self, question:str, context:str, answers:[]):
        rnd = torch.rand(len(answers))
        out = torch.softmax(rnd, 0)
        out.requires_grad = True
        attn = torch.rand(len(question.split()))
        attn.requires_grad = True
        return out, attn