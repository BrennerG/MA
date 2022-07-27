import torch
import torch.nn as nn
import torch.nn.functional as F

# This is an extension of the RandomClassifier.
# In addition to labeling randomly, this Classifier returns a randomized fake attention vector.
class RandomAttentionClassifier(nn.Module):

    def __init__(self, seed:int):
        super().__init__()
        self.TYPE = "RandomAttentionClassifier"
        self.rationality = 'soft'
        self.seed = seed
        self.lin = TestModule()
        torch.manual_seed(self.seed)

    def forward(self, question:str, context:str, answers:[], label):
        label = None # don't cheat
        rnd = torch.rand(len(answers))
        out = torch.softmax(rnd, 0)
        out.requires_grad = True
        attn = torch.rand(len(question))
        attn.requires_grad = True
        return out, attn

# TODO add option to return hard rationales! (or write another RND classifier for that~)


class TestModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(150, 5)
    
    def forward(self, x):
        x = x.view(1000,-1)
        x = self.linear(x)
        return x