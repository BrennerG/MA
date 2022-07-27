import torch
import torch.nn as nn
import torch.nn.functional as F

# A Classifier that answers questions randomly, utilizing softmax on random numbers!
# Does not provide any form fake attention vectors!
class RandomClassifier(nn.Module):

    def __init__(self, seed:int):
        super().__init__()
        self.TYPE = "RandomClassifier"
        self.seed = seed
        self.lin = nn.Linear(3, 1) # an alibi linear layer - this is not actually used!
        torch.manual_seed(self.seed)

    def forward(self, question:str, context:str, answers:[]):
        rnd = torch.rand(len(answers))
        out = torch.softmax(rnd, 0)
        out.requires_grad = True
        return out, None # output, attn