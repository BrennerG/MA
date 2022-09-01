import stanza
from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt

from datasets import load_dataset
from data.locations import LOC

def run():
    num_samples = 3
    cose = load_dataset(LOC['cose_huggingface'])
    dataset = cose['train']
    nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse")

    # LOOP
    edge_indices = []
    tokens = []
    for i, sample in tqdm(enumerate(dataset)):
        if i > num_samples: break
        for answer in sample['answers']:
            inp = f"{sample['question']} {answer}"
            doc =  nlp(inp)
            parsed = [word for sent in doc.sentences for word in sent.words]
            edges = [(x.id-1, x.head-1) for x in parsed] # 0 is first tokens, -1 is root now
            edge_indices.append(edges)
            tokens.append([x.text for x in parsed])

    # LOOK AT GRAPHs
    i = 1
    G = nx.Graph(edge_indices[i])
    nx.set_node_attributes(G, {k:{'text':v} for k,v in list(zip(range(len(tokens[i])), tokens[i]))})
    labels = nx.get_node_attributes(G, 'text') 
    nx.draw(G, labels=labels, with_labels=True)
    plt.show()

    # DONE
    return None