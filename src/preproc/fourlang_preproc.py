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
    def parse(self, dataset, num_samples, split, qa_join, **kwargs):
        use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else False
        use_existing_concept_ids = kwargs['use_existing_concept_ids'] if 'use_existing_concept_ids' in kwargs else False
        max_num_nodes = kwargs['max_num_nodes'] if 'max_num_nodes' in kwargs else None
        expand = kwargs['expand'] if 'expand' in kwargs else None

        edges_path = LOC['4lang_parses'] + f'cose_{split}_{str(num_samples)}_{qa_join}.json'
        if os.path.exists(edges_path) and use_cache:
            print(f'4Lang_Parsing: Accessing cached file: {edges_path}')
            with open(edges_path) as f:
                edges = json.load(f)
        else:
            edges = self.extract_edges(dataset=dataset, num_samples=num_samples, qa_join=qa_join, use_existing_concept_ids=use_existing_concept_ids, max_num_nodes=max_num_nodes, expand=expand)
            if use_cache:
                # save edges
                with open(edges_path, 'w') as outfile:
                    json.dump(edges, outfile)
        
        return edges

    def extract_edges(self, dataset, num_samples=-1, qa_join='none', use_existing_concept_ids=False, max_num_nodes=None, expand=None):
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
                largest_parse = self.fourlang_parse(qa, max_num_nodes=max_num_nodes, expand=expand)

                # are there any edges?
                if len(largest_parse.edges) == 0:
                   largest_parse = self.search_for_possible_graph(qa_tokenized, largest_parse, max_num_nodes=max_num_nodes, expand=expand)
                
                # fallback method: use question context to build graph
                if largest_parse == None:
                    largest_parse = self.fourlang_parse(sample['context'], max_num_nodes=max_num_nodes, expand=expand)
                
                # somehow the ids given by 4Lang reset during explainability eval
                # so this clause is rather specifically for expl.eval to use the established concept ids
                if use_existing_concept_ids:
                    names = nx.get_node_attributes(largest_parse, 'name')
                    old_ids = []
                    for x in names.values():
                        if x in self.concept2id:
                            old_ids.append(self.concept2id[x])
                        else:
                            pos = max(self.id2concept)+1
                            self.id2concept[pos] = x
                            self.concept2id[x] = pos
                            old_ids.append(pos)
                    map_to_existing = dict(zip(list(largest_parse.nodes), old_ids))
                    largest_parse = nx.relabel_nodes(largest_parse, map_to_existing)

                assert len(largest_parse.edges) > 0

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
    
    def search_for_possible_graph(self, qa_tokenized, parse, max_num_nodes, expand):
        running_tokens = qa_tokenized.copy()
        while len(running_tokens) > 0 and len(parse.edges) == 0:
            running_tokens = running_tokens[:-1]
            if len(running_tokens) == 0: return None
            parse = self.fourlang_parse(" ".join(running_tokens), max_num_nodes, expand)
            if len(parse.edges) > 0: return parse
        return None
    
    def dynamic_4lang_parse(self, qa, max_num_nodes=200):
        running_depth = 0
        fails = 0
        initial_parse = list(self.tfl(qa, depth=running_depth, substitute=False))
        initial_largest_parse = initial_parse[np.argmax([len(x) for x in initial_parse])]
        parses = [initial_largest_parse]

        while(len(parses[-1].nodes)<=max_num_nodes):
            parse = list(self.tfl(qa, depth=running_depth, substitute=False))
            largest_parse = parse[np.argmax([len(x) for x in parse])]
            parses.append(largest_parse)
            running_depth += 1
            if len(largest_parse.edges) < 1 or len(largest_parse.nodes) < 1:
                return initial_parse
        
        return parses[-2]
    
    def fourlang_parse(self, qa, max_num_nodes=None, expand=None):
        if max_num_nodes != None:
            return self.dynamic_4lang_parse(qa, max_num_nodes)
        elif expand != None:
            parse = list(self.tfl(qa, depth=expand, substitute=False))
            largest_parse = parse[np.argmax([len(x) for x in parse])]
            return largest_parse