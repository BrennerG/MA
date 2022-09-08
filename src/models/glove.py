import numpy as np
from tqdm import tqdm
import torch
from tests.ud_preproc import UDParser


class GloveEmbedder():

    def __init__(self, file_location='data/glove/glove.840B.300d.txt'):
        self.ud_parser = UDParser()
        self.embeddings_dict = {}
        self.dim = 300
        with open(file_location) as f:
            for line in tqdm(f, desc="loading glove..."):
                values = line.split()
                word = ''.join(values[:-self.dim])
                vector = np.asarray(values[-self.dim:], "float32")
                self.embeddings_dict[word] = vector
    
    def __call__(self, sentence:str, unk_procedure='zero'):
        return self.embed(sentence, unk_procedure=unk_procedure)
    
    def get(self, word:str):
        if word in self.embeddings_dict:
            return self.embeddings_dict[word]
        else:
            return None
    
    def embed(self, sentence:str, unk_procedure='random'):
        tokens = self.ud_parser.tokenize(sentence)
        emb = np.zeros((len(tokens), self.dim))
        for i,t in enumerate(tokens):
            if t in self.embeddings_dict:
                emb[i] = self.get(t)
            elif unk_procedure=='zero':
                continue
            elif unk_procedure=='random':
                emb[i] = np.random.rand(self.dim)
            else:
                raise AttributeError(f'unk_procedure {unk_procedure} is unknown!')
        return torch.Tensor(emb)