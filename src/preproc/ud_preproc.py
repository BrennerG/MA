import stanza
from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json

from datasets import load_dataset
from data.locations import LOC
from preproc.graph_preproc import GraphPreproc


class UDParser(GraphPreproc):

    def __init__(self, params, processors="tokenize,mwt,pos,lemma,depparse"):
        super().__init__()
        self.params = params
        self.ud_parser = stanza.Pipeline(lang='en', processors=processors, use_gpu=params['use_cuda'])
        self.tokenizer = stanza.Pipeline(lang='en', processors="tokenize", use_gpu=params['use_cuda'])
        self.root_token = '[ROOT]'

    def parse(self, dataset, num_samples=-1, split='train', qa_join='none', use_cache=True):
        file_path = LOC['ud_parses'] + f'cose_{split}_{str(num_samples)}_{qa_join}.json'
        if os.path.exists(file_path) and use_cache:
            print(f'UD_Parsing: Accessing cached file: {file_path}')
            with open(file_path) as f:
                edges = json.load(f)
        else:
            edges = self.extract_edges(dataset=dataset, num_samples=num_samples, qa_join=qa_join)
            if use_cache:
                with open(file_path, 'w') as outfile:
                    json.dump(edges, outfile)
        return edges

    def extract_edges(self, dataset, num_samples=-1, qa_join='none'):
        edge_indices = []

        for i, sample in enumerate(tqdm(dataset, desc='ud-parsing cose...')):
            grouped_edges = [] # 5 graphs per sample (=QA-pair)
            if num_samples > 0 and i >= num_samples: break # sample cut-off
            for answer in sample['answers']:
                # parse through stanza & extract UD edges
                if '?' in sample['question']: # TODO do this in dataset class?
                    doc = self.ud_parser(f"{sample['question']} {answer}")
                else:
                    doc = self.ud_parser(f"{sample['question']} ? {answer}")
                parsed = [word for sent in doc.sentences for word in sent.words] # stanza parse
                edges = [(x.head-1, x.id-1) for x in parsed] # 0 is first tokens, -1 is root now
                edges = [(a,b) for (a,b) in edges if a!=-1 and b!=-1]  # remove edges to root
                # Q/A GRAPH JOINING METHOD
                if qa_join=='a-to-all': # connect answer to all other nodes 
                    tokens = [x.text for x in parsed]
                    answer_nodes = list(range(len(tokens)))[tokens.index('?')+1:]
                    for an in answer_nodes:
                        edges.extend(
                            zip(range(len(tokens)), [an]*len(tokens))
                        )
                elif qa_join=='to-root':
                    tokens = [x.text for x in parsed]
                    edges = [(x.head-1, x.id-1) for x in parsed] # 0 is first token, -1 is root now
                    edges = [(a,b) if a != -1 else (len(tokens),b) for (a,b) in edges] # put root at the end of the tokens
                    answer_nodes = list(range(len(tokens)))[tokens.index('?')+1:]
                    for an in answer_nodes:
                        edges.append((an, len(tokens)))
                elif qa_join=='none': # dont join q and a
                    pass
                else:
                    raise AttributeError(f'Unknown qa_joining method: {qa_join}')
                grouped_edges.append(list(set(edges)))
                # TODO create post_process method for the following:
                # TODO what about opposit edges? e.g. (1,3) & (3,1)? (rm via nx?)
                # TODO also rm self edges? (rm via nx?)
                # TODO also rm edges to/from '?' (might collide with BERT [SEP] later)
            edge_indices.append(grouped_edges)

        return edge_indices

    # tokenizes the same way as extract_edges does!
    def tokenize(self, sentence:str):
        doc = self.tokenizer(sentence)
        parsed = [word for sent in doc.sentences for word in sent.words] # stanza parse
        tokens = [x.text for x in parsed]
        if 'qa_join' in self.params and self.params['qa_join'] == 'to-root':
            tokens.append(self.root_token)
        return tokens

    # show plot for graph!
    def show(self, edge_index, tokens):
        G = nx.Graph(edge_index)
        nx.set_node_attributes(G, {k:{'text':v} for k,v in list(zip(range(len(tokens)), tokens))})
        labels = nx.get_node_attributes(G, 'text') 
        nx.draw(G, labels=labels, with_labels=True)
        plt.show()