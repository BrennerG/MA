from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json
import stanza

from datasets import load_dataset
from tuw_nlp.grammar.text_to_4lang import TextTo4lang

from data.locations import LOC
from preproc.graph_preproc import GraphPreproc


# TODO do qa_joining centrally or is this not possible?

class FourLangParser(GraphPreproc):

    def __init__(self, params:{}):
        super().__init__()
        self.params = params
        self.tfl = TextTo4lang("en", "en_nlp_cache")
        self.tokenizer = stanza.Pipeline(lang='en', processors="tokenize", use_gpu=params['use_cuda'])
        self.id2concept = {}
        self.concept2id = {}
    
    # __call__
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
        edges = []
        maps = []
        for i, sample in enumerate(tqdm(dataset, desc='4lang-parsing cose...')):
            # 5 graphs per sample (=QA-pair)
            grouped_edges = [] 
            grouped_maps = []
            if num_samples > 0 and i >= num_samples: break # sample cut-off
            for answer in sample['answers']:
                # TODO current joining method: 'none'
                qa = f"{sample['question']} {answer}"
                qa_tokenized = self.tokenize(qa)
                # TODO expansion mechanism here (set tfl depth>1)
                parse = list(self.tfl(qa, depth=1, substitute=False))
                # save mapping from qa order to 4lang ids
                qa_to_4lang_map = []
                names = nx.get_node_attributes(parse[0], 'name')
                for qai, tok in enumerate(qa_tokenized):
                    for i,w in names.items():
                        if tok == w or tok.lower() == w.lower():
                            qa_to_4lang_map.append((qai,i))
                # append
                grouped_edges.append(list(parse[0].edges))
                grouped_maps.append(qa_to_4lang_map)
                # save voc
                for i,x in names.items():
                    if i not in self.id2concept:
                        self.id2concept[i] = x
                        self.concept2id[x] = i
                    elif i in self.id2concept and self.id2concept[i] != x:
                        raise LookupError('ERR: inconsistent voc!')
                pass

            edges.append(grouped_edges)
            maps.append(grouped_maps)

        return edges, maps
    
    def tokenize(self, sentence:str):
        doc = self.tokenizer(sentence)
        parsed = [word for sent in doc.sentences for word in sent.words] # stanza parse
        tokens = [x.text for x in parsed]
        if 'qa_join' in self.params and self.params['qa_join'] == 'to-root':
            tokens.append(self.root_token)
        return tokens

    def show(self, edge_index, tokens):
        raise NotImplementedError()

    def tokens_from_nx(self, graph:nx.Graph): # TODO this method needed?
        # return [x[1]['name'] for x in parse[1].nodes.data()] # TODO check order?
        return None
