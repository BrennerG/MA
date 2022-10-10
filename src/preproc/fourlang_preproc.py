from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json

from datasets import load_dataset
from data.locations import LOC
from preproc.graph_preproc import GraphPreproc

from tuw_nlp.grammar.text_to_4lang import TextTo4lang


# TODO do qa_joining centrally or is this not possible?

class FourLangParser(GraphPreproc):

    def __init__(self, params:{}):
        super().__init__()
        self.tfl = TextTo4lang("en", "en_nlp_cache")
    
    def parse(self, dataset, num_samples, split, qa_join, use_cache=True):
        file_path = LOC['4lang_parses'] + f'cose_{split}_{str(num_samples)}_{qa_join}.json'
        if os.path.exists(file_path) and use_cache:
            print(f'4Lang_Parsing: Accessing cached file: {file_path}')
            with open(file_path) as f:
                edges = json.load(f)
        else:
            edges = self.extract_edges(dataset=dataset, num_samples=num_samples, qa_join=qa_join)
            if use_cache:
                with open(file_path, 'w') as outfile:
                    json.dump(edges, outfile)
        return edges

    def extract_edges(self, dataset, num_samples=-1, qa_join='none'):
        for i, sample in enumerate(tqdm(dataset, desc='4lang-parsing cose...')):
            grouped_edges = [] # 5 graphs per sample (=QA-pair)
            if num_samples > 0 and i >= num_samples: break # sample cut-off
            for answer in sample['answers']:
                qa = f"{sample['question']} {answer}"
                parse = list(self.tfl(qa, depth=1, substitute=False))
                pass # TODO CURRENT

        return None
    
    def tokenize(self, sentence:str):
        raise NotImplementedError()

    def show(self, edge_index, tokens):
        raise NotImplementedError()

    def tokens_from_nx(self, graph:nx.Graph): # TODO this method needed?
        # return [x[1]['name'] for x in parse[1].nodes.data()] # TODO check order?
        return None
