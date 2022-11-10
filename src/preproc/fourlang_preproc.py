from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json
import stanza
import numpy as np

from datasets import load_dataset
from tuw_nlp.grammar.text_to_4lang import TextTo4lang

from data.locations import LOC
from preproc.graph_preproc import GraphPreproc


# TODO do qa_joining centrally or is this not possible?

class FourLangParser(GraphPreproc):

    def __init__(self, params:{}, use_cache=True):
        super().__init__()
        self.params = params
        self.tfl = TextTo4lang("en", "en_nlp_cache")
        self.tokenizer = stanza.Pipeline(lang='en', processors="tokenize", use_gpu=params['use_cuda'])
        if os.path.exists(LOC['4L_concept2id']) and use_cache:
            with open(LOC['4L_concept2id']) as f:
                self.concept2id = json.load(f)
        else:
            self.concept2id = {}
        if os.path.exists(LOC['4L_id2concept']) and use_cache:
            with open(LOC['4L_id2concept']) as f:
                self.id2concept = json.load(f)
                self.id2concept = {int(k):v for k,v in self.id2concept.items()}
        else:
            self.id2concept = {}
        pass
    
    # __call__
    def parse(self, dataset, num_samples, split, qa_join, use_cache=True):
        edges_path = LOC['4lang_parses'] + f'cose_{split}_{str(num_samples)}_{qa_join}.json'
        if os.path.exists(edges_path) and use_cache:
            print(f'4Lang_Parsing: Accessing cached file: {edges_path}')
            with open(edges_path) as f:
                edges = json.load(f)
        else:
            edges = self.extract_edges(dataset=dataset, num_samples=num_samples, qa_join=qa_join)
            if use_cache:
                # save edges
                with open(edges_path, 'w') as outfile:
                    json.dump(edges, outfile)
        
        return edges

    def extract_edges(self, dataset, num_samples=-1, qa_join='none'):
        edges = []
        maps = []
        concepts = []
        for i, sample in enumerate(tqdm(dataset, desc='4lang-parsing cose...')):
            # 5 graphs per sample (=QA-pair)
            grouped_edges = [] 
            grouped_maps = []
            grouped_concepts = []
            if num_samples > 0 and i >= num_samples: break # sample cut-off
            for answer in sample['answers']:
                # TODO current joining method: 'none'
                qa = f"{sample['question']} {answer}"
                qa_tokenized = self.tokenize(qa)
                # TODO expansion mechanism here (set tfl depth>1)
                parse = list(self.tfl(qa, depth=1, substitute=False))

                # mapping from nodes to og tokens
                nodes_to_qa_tokens = []
                names = nx.get_node_attributes(parse[0], 'name')
                for i,x in names.items():
                    if x in qa_tokenized:
                        nodes_to_qa_tokens.append(qa_tokenized.index(x))
                    else:
                        nodes_to_qa_tokens.append(None)
                # append
                grouped_edges.append(list(parse[0].edges))
                grouped_maps.append(nodes_to_qa_tokens)
                grouped_concepts.append(list(names.values()))
                # save voc
                for i,x in names.items():
                    if x not in self.concept2id:
                        if i in self.id2concept and self.id2concept[i] != x:
                            pos = max(self.id2concept)+1
                        else: 
                            pos = i
                        self.id2concept[pos] = x
                        self.concept2id[x] = pos

            edges.append(grouped_edges)
            maps.append(grouped_maps)
            concepts.append(grouped_concepts)

        return edges, maps, concepts
    
    def tokenize(self, sentence:str):
        doc = self.tokenizer(sentence)
        parsed = [word for sent in doc.sentences for word in sent.words] # stanza parse
        tokens = [x.text for x in parsed]
        if 'qa_join' in self.params and self.params['qa_join'] == 'to-root':
            tokens.append(self.root_token)
        return tokens

    def show(self, edge_index, tokens):
        raise NotImplementedError()
    
    def save_concepts():
        # save 4L dicts
        assert self.concept2id != {} and self.id2concept != {}
        with open(dicts_path['concept2id'], 'w') as outfile:
            json.dump(self.concept2id, outfile)
        with open(dicts_path['id2concept'], 'w') as outfile:
            json.dump(self.id2concept, outfile)