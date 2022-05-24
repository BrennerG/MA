import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO add option to return hard rationales!
# TODO transform into Rationales, Annotations, probably in a model_util class or sth?
class RandomAttentionClassifier(nn.Module):

    def __init__(self, seed:int):
        super().__init__()
        self.TYPE = "RandomAttentionClassifier"
        self.rationality = 'soft'
        self.seed = seed
        self.lin = nn.Linear(3, 1)
        torch.manual_seed(self.seed)

    def forward(self, question:str, context:str, answers:[]):
        rnd = torch.rand(len(answers))
        out = torch.softmax(rnd, 0)
        out.requires_grad = True
        attn = torch.rand(len(question))
        attn.requires_grad = True
        return out, attn