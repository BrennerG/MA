import stanza
from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt
import os.path
import json

from datasets import load_dataset
from data.locations import LOC
            

def parse(dataset, num_samples=-1, split='train'):
    file_path = LOC['ud_parses'] + f'cose_{split}_{str(num_samples)}.json'
    if os.path.exists(file_path):
        with open(file_path) as f:
            edges = json.load(f)
    else:
        edges = extract_edges(dataset=dataset, num_samples=num_samples)
        with open(file_path, 'w') as outfile:
            json.dump(edges, outfile)
    return edges

def extract_edges(dataset, num_samples=-1):
    nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse")
    edge_indices = []

    for i, sample in enumerate(tqdm(dataset, desc='UD-parsing cose...')):
        grouped_edges = [] # 5 graphs per sample (=QA-pair)
        if num_samples > 0 and i >= num_samples: break # sample cut-off
        for answer in sample['answers']:
            # parse through stanza & extract UD edges
            doc =  nlp(f"{sample['question']} {answer}")
            parsed = [word for sent in doc.sentences for word in sent.words] # stanza parse
            edges = [(x.id-1, x.head-1) for x in parsed] # 0 is first tokens, -1 is root now
            edges = [(a,b) for (a,b) in edges if a!=-1 and b!=-1]  # remove edges to root
            # connect answer to all other nodes 
            tokens = [x.text for x in parsed]
            answer_nodes = list(range(len(tokens)))[tokens.index('?')+1:]
            for an in answer_nodes:
                edges.extend(
                    zip(range(len(tokens)), [an]*len(tokens))
                )
            # yield
            grouped_edges.append(list(set(edges)))
            # TODO create post_process method for the following:
            # TODO what about opposit edges? e.g. (1,3) & (3,1)? (rm via nx?)
            # TODO also rm self edges? (rm via nx?)
            # TODO also rm edges to/from '?' (might collide with BERT [SEP] later)
        edge_indices.append(grouped_edges)

    return edge_indices

# show plot for graph!
def show(edge_index, tokens):
    G = nx.Graph(edge_index)
    nx.set_node_attributes(G, {k:{'text':v} for k,v in list(zip(range(len(tokens)), tokens))})
    labels = nx.get_node_attributes(G, 'text') 
    nx.draw(G, labels=labels, with_labels=True)
    plt.show()