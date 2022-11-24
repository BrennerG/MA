from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json
import stanza
import numpy as np
import random

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
                parse = list(self.tfl(qa, depth=1, substitute=False)) # TODO expansion mechanism here (set tfl depth>1) & dynamic num_node capping
                largest_parse = parse[np.argmax([len(x) for x in parse])]

                # are there any edges?
                if len(largest_parse.edges) == 0:
                    print("WARNING: 4L could not parse this QA. searching... (this may take a while)")
                    running_tokens = qa_tokenized.copy()
                    parse = largest_parse.copy()
                    while len(running_tokens) > 0 and len(parse.edges) == 0:
                        running_tokens = running_tokens[:-1]
                        parse = list(self.tfl(" ".join(running_tokens), depth=1, substitute=False))[0] # TODO expansion mechanism here (set tfl depth>1) & dynamic num_node capping
                    largest_parse = parse

                # get mapping from nodes in order to qa_tokens
                names = nx.get_node_attributes(largest_parse, 'name')
                nodes_to_qa_tokens = self.map_nodes_to_og_tokens(names, qa_tokenized)

                # correct dict ids
                ordered_names = {x:names[x] for x in sorted(names.keys())}
                corr_map = dict(zip(ordered_names, range(len(ordered_names))))
                relative_edges = [(corr_map[x], corr_map[y]) for x,y in largest_parse.edges]

                # append
                grouped_edges.append(list(largest_parse.edges))
                grouped_maps.append(nodes_to_qa_tokens)
                grouped_concepts.append(list(names.values()))
            
                # save voc
                for n,x in names.items():
                    if x not in self.concept2id:
                        if n in self.id2concept and self.id2concept[n] != x:
                            pos = max(self.id2concept)+1
                        else: 
                            pos = n
                        self.id2concept[pos] = x
                        self.concept2id[x] = pos

            assert all([len(x) for x in grouped_edges]), 'ASSERTION ERROR: a 4L qa_graph has 0 edges D:' # check if #edges are >0 for all qa_graphs!
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
    
    def save_concepts(self):
        # save 4L dicts
        assert self.concept2id != {} and self.id2concept != {}
        with open(LOC['4L_concept2id'], 'w') as outfile:
            json.dump(self.concept2id, outfile)
        with open(LOC['4L_id2concept'], 'w') as outfile:
            json.dump(self.id2concept, outfile)

    def map_nodes_to_og_tokens(self, names:dict, qa_tokenized:[]):
        nodes_to_qa_tokens = []
        for n,x in names.items():
            if x in qa_tokenized:
                nodes_to_qa_tokens.append(qa_tokenized.index(x))
            else:
                nodes_to_qa_tokens.append(None)
        return nodes_to_qa_tokens