import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class BagOfWordsClassifier():

    def __init__(self, parameters:{}):
        self.TYPE = "BagOfWordsClassifier"

    def train(self, ds:Dataset()):
        pass

    def predict(self, x):
        if isinstance(x, Dataset):
            return None
        else:
            raise NotImplemented