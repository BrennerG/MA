import stanza
from tqdm import tqdm
import networkx as nx
import matplotlib.pylab as plt

from datasets import load_dataset
from data.locations import LOC
            
            
def join_qa(question, answer, mode='none'):
    if mode == 'none':
        return f"{question} {answer}"
    elif mode == 'wh-sub': # TODO this is obviously naive, since UD parsing relies on well-formed natural language
        WHs = ['what', 'where', 'why'] # TODO check how many of them actually contain WH-questions
        tokens = [q.lower() for q in question.split()]
        wh_positions = [tokens.index(w) if w in tokens else None for w in WHs]
        assert any([x!=None for x in wh_positions]) # TODO use mode='none' for these examples (if relevant)
        for wh in wh_positions:
            if wh != None: # TODO this returns first non-None position
                tokens[wh] = answer
                return " ".join(tokens)
    else:
        raise AttributeError(f"joining mode '{mode}' unknown!")

def parse_cose(num_samples=-1, split='train', join_mode='wh-sub'):
    num_samples = 2
    cose = load_dataset(LOC['cose_huggingface'])
    dataset = cose[split]
    nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse")

    # LOOP
    edge_indices = []
    for i, sample in tqdm(enumerate(dataset)):
        if num_samples > 0 and i > num_samples: break
        for answer in sample['answers']:
            inp = join_qa(sample['question'], answer, mode=join_mode)
            doc =  nlp(inp)
            parsed = [word for sent in doc.sentences for word in sent.words]
            edges = [(x.id-1, x.head-1) for x in parsed] # 0 is first tokens, -1 is root now
            root_removed_edges = [(a,b) for (a,b) in edges if a!=-1 and b!=-1] # TODO leave it like this? (or use 'to_root' as node attr!)
            edge_indices.append(root_removed_edges) 

    # DONE
    return edge_indices

# look at graphs
def show(edge_index, tokens):
    G = nx.Graph(edge_index)
    nx.set_node_attributes(G, {k:{'text':v} for k,v in list(zip(range(len(tokens[i])), tokens[i]))})
    labels = nx.get_node_attributes(G, 'text') 
    nx.draw(G, labels=labels, with_labels=True)
    plt.show()