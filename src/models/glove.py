import os
import numpy as np
from tqdm import tqdm
import torch
from preproc.ud_preproc import UDParser
from data.locations import LOC
import json


class GloveEmbedder():

    def __init__(self, params:{}, file_location=LOC['glove_embedding'], use_cached=True):
        self.unk_procedure = 'random' # TODO put in params?
        self.ud_parser = UDParser(params=params)
        self.embeddings_dict = {}
        self.cached_dict = {}
        self.dim = 300

        if use_cached:
            # look for file
            if not os.path.exists(LOC['glove_cache']):
                print('WARNING: could not find cached glove file. loading complete glove embedding (this can take 2-5 minutes)')
                self.embeddings_dict = self.load_glove_dict(file_location=file_location)
            else:
                with open(LOC['glove_cache']) as f:
                    print(f"Loaded cached glove vocabulary from {LOC['glove_cache']}")
                    self.embeddings_dict = json.load(f)
        else:
            self.embeddings_dict = self.load_glove_dict(file_location=file_location)

           
    def load_glove_dict(self, file_location):
        embeddings_dict = {}
        # default mode
        with open(file_location) as f:
            for line in tqdm(f, desc="loading glove..."):
                values = line.split()
                word = ''.join(values[:-self.dim])
                vector = np.asarray(values[-self.dim:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict
 
    
    def __call__(self, sentence:str):
        return self.embed(sentence)
    
    def get(self, word:str):
        if word in self.embeddings_dict:
            emb = self.embeddings_dict[word]
            if not isinstance(emb, np.ndarray): # case for cached
                self.cached_dict[word] = list(emb)
                return np.array(emb)
            else:
                self.cached_dict[word] = emb.tolist()
                return emb
        else:
            print(f"WARNING token: '{word}' not in glove vocabulary (unk_procedure='{self.unk_procedure}'")
            return None
    
    def embed(self, sentence:str):
        tokens = self.ud_parser.tokenize(sentence)
        emb = np.zeros((len(tokens), self.dim))
        for i,t in enumerate(tokens):
            if t in self.embeddings_dict:
                emb[i] = self.get(t)
            elif self.unk_procedure=='zero':
                continue
            elif self.unk_procedure=='random':
                emb[i] = np.random.rand(self.dim)
            else:
                raise AttributeError(f'unk_procedure {unk_procedure} is unknown!')
        return torch.Tensor(emb)