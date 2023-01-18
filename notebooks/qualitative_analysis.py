import json

import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network
import streamlit.components.v1 as components


# PARAMS
EXPERIMENT = "default"
MA_FILE = f"notebooks/data//viz_data.json"
RND_SAMPLE_FILE = "notebooks/data/random_quali_select_from_test.json"
OG_FILE = f"notebooks/data/SAVED_DIC.json"

# METHODS
@st.cache(allow_output_mutation=True)
def load_ma_data(datafile, rnd_sample_file=RND_SAMPLE_FILE):
    # load random file
    with open(rnd_sample_file, 'r') as file:
        rnd_sample_ids = json.load(file)
    rnd_sample_ids = rnd_sample_ids[:10] # TODO change (first 10 samples are for debugging)

    # load samples
    with open(datafile, 'r') as file:
        raw_data = json.load(file)
    
    # apply random selection
    rnd_sample_idx = [x['id'] for x in raw_data['statement_data']]
    rnd_sample_idx = [rnd_sample_idx.index(x) for x in rnd_sample_ids]

    # separate concepts and raw_data
    id2concept = raw_data.pop('4L_id2concept')
    data = {k:[x for i,x in enumerate(v) if i in rnd_sample_idx] for k,v in raw_data.items()}

    # Pandas DataFrame
    #df = pd.DataFrame(data)
    ## extend with statement_data data
    #df_extension = {k:[] for (k,v) in data['statement_data'][0].items()}
    #for sample in data['statement_data']:
    #    for k,v in sample.items():
    #        df_extension[k].append(v)
    #df.drop('statement_data', inplace=True, axis=1) 
    ## extend df's rows
    #for k,v in df_extension.items():
    #    df[k] = v
    #df.set_index('id', inplace=True)

    #extend with statement_data data
    extension = {k:[] for (k,v) in data['statement_data'][0].items()}
    for sample in data['statement_data']:
        for k,v in sample.items():
            extension[k].append(v)
    data.pop('statement_data')

    # extend data's entries
    for k,v in extension.items():
        data[k] = v

    return rnd_sample_ids, data, id2concept

@st.cache()
def load_og_data(datafile, rnd_sample_file=RND_SAMPLE_FILE):
    # load random file
    with open(rnd_sample_file, 'r') as file:
        rnd_sample_ids = json.load(file)
    rnd_sample_ids = rnd_sample_ids[:10] # TODO change (first 10 samples are for debugging)

    # load samples
    with open(datafile, 'r') as file:
        raw_data = json.load(file)

    # load statement data
    STATEMENT_FILE = 'notebooks/data/test.statement.jsonl'
    statement_data = []
    with open(STATEMENT_FILE, 'r') as file:
        for line in file:
            statement_data.append(json.loads(line))

    raw_data['statement_data'] = statement_data
    rnd_sample_idx = [x['id'] for x in raw_data['statement_data']]
    rnd_sample_idx = [rnd_sample_idx.index(x) for x in rnd_sample_ids]

    raw_data.pop('id2concept')
    st.write(raw_data.keys())
    data = {k:[x for i,x in enumerate(v) if i in rnd_sample_idx] for k,v in raw_data.items()}

    # load id2concept
    CONCEPT_FILE = 'notebooks/data/concept.txt'
    with open(CONCEPT_FILE) as _file_:
        id2concept = [line for line in _file_]

    return data, id2concept

def create_og_graph(data, option_id):
    C = np.argmax(data['logits'][option_id]) # choice
    st.write(f"choice = {C} : {data['statement_data'][option_id]['question']['choices'][C]['text']}")

    # nodes
    concept_ids = data['concept_ids'][option_id][C]
    concept_names = [id2concept[x] for x in concept_ids]
    node_type_ids = data['node_type_ids'][option_id][C]
    node_scores = [x[0] for x in data['node_scores'][option_id][C]]
    colors = ['red', 'green', 'blue', 'purple']
    colorcode = [colors[x] for x in node_type_ids]
    N = len(concept_ids)

    # edges
    raw_edges = data['edge_index_orig'][option_id][C]
    edges = [(raw_edges[0][i], raw_edges[1][i]) for i in range(len(raw_edges[0]))]
    E = len(edges)

    # graph
    st.write(f"N={N}, E={E}")
    net = Network(notebook=True)
    net.add_nodes(
        range(N),
        value = node_scores,
        title = concept_names,
        label = concept_names,
        color = colorcode
    )
    for e in edges: net.add_edge(*e)

    return net

def create_4L_graph(sample):

    # choice
    logits = sample['logits']
    choice = np.argmax(logits)
    st.write(f"choice = {choice} : { sample['answers'][choice]}")

    # nodes
    node_scores = sample['node_scores'][choice]
    if 'Z_VEC' in node_scores: node_scores.pop('Z_VEC')
    concept_names = sample['4L_concept_names'][choice]
    assert len(node_scores) == len(concept_names)
    N = len(node_scores)

    # edges
    edges = sample['4L_edges'][choice][0]
    edge_types = sample['4L_edges'][choice][1]
    E = len(edges)

    # colors
    flq_map = SAMPLE['4L_map'][choice]
    colorcode = ['red' if x != None else 'blue' for x in flq_map]
    
    # construct graph
    st.write(f"N={N}, E={E}")
    net = Network(notebook=True)
    net.add_nodes(
        range(N),
        value=list([node_scores[x] for x in concept_names]),
        title=concept_names,
        label=concept_names,
        color=colorcode
    )
    for e in edges: net.add_edge(*e)

    return net

def show_graph(net, jiggle=True):
    graph_path = 'notebooks/tmp/fl_graph.html'
    net.toggle_physics(jiggle)
    net.show(graph_path)
    HtmlFile = open(graph_path, 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=610)

################## ################## ################## ##################
################## ############### MAIN ################ ##################
################## ################## ################## ##################

st.set_page_config(layout="wide")
sel_ids, data, id2concept = load_ma_data(MA_FILE)

################## ################## SIDEBAR ################## ##################

with st.sidebar:
    option = st.selectbox("which sample?", sel_ids)
    option_id = sel_ids.index(option)
    st.write(option_id)

################## ################## HEADER ################## ##################

with st.container():
    #st.write(pd.DataFrame(data))
    SAMPLE = {k:v[option_id] for k,v in data.items()}
    st.title(SAMPLE['question']) # TODO use real question here
    st.write({k:v for k,v in SAMPLE.items() if k in ['question', 'answers', 'label']})

################## ################## COLS ################## ##################

LEFT_C, RIGHT_C = st.columns(2, gap='medium')

with LEFT_C:
    st.subheader("cpnet + QA-GNN")
    data, id2concept = load_og_data(OG_FILE, RND_SAMPLE_FILE)
    net = create_og_graph(data,option_id)
    show_graph(net)
    
################## ################## SEPARATOR ################## ##################

with RIGHT_C:
    st.subheader("4Lang + QA-GNN")
    net = create_4L_graph(SAMPLE)
    show_graph(net)

st.balloons()