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
FILE = f"../data/experiments/{EXPERIMENT}/viz_data.json"
RND_SAMPLE_FILE = "../random_quali_select_from_test.json"

# METHODS
@st.cache(allow_output_mutation=True)
def load_data(datafile, rnd_sample_file=RND_SAMPLE_FILE):
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

################## ################## ################## ##################
################## ############### MAIN ################ ##################
################## ################## ################## ##################

st.set_page_config(layout="wide")
sel_ids, data, id2concept = load_data(FILE)

################## ################## SIDEBAR ################## ##################

with st.sidebar:
    option = st.selectbox("which sample?", sel_ids)
    option_id = sel_ids.index(option)
    st.write(option_id)

################## ################## HEADER ################## ##################

with st.container():
    st.write(pd.DataFrame(data))
    SAMPLE = {k:v[option_id] for k,v in data.items()}
    st.title(SAMPLE['question']) # TODO use real question here
    st.write({k:v for k,v in SAMPLE.items() if k in ['question', 'answers', 'label']})

################## ################## COLS ################## ##################

LEFT_C, RIGHT_C = st.columns(2, gap='medium')

with LEFT_C:
    st.subheader("cpnet + QA-GNN")
    st.write("Nothing here yet...")

with RIGHT_C:
    st.subheader("4Lang + QA-GNN")

    # def draw_graph(..):

    # choice
    logits = SAMPLE['logits']
    choice = np.argmax(logits)

    # nodes
    node_scores = SAMPLE['node_scores'][choice]
    if 'Z_VEC' in node_scores: node_scores.pop('Z_VEC')
    concept_names = SAMPLE['4L_concept_names'][choice]
    assert len(node_scores) == len(concept_names)
    N = len(node_scores)
    concept_names = SAMPLE['4L_concept_names'][choice]

    # edges
    edges = SAMPLE['4L_edges'][choice][0]
    E = len(edges)
    edge_types = SAMPLE['4L_edges'][choice][1]

    # colors
    flq_map = SAMPLE['4L_map'][choice]
    node_type_ids = SAMPLE['node_type_ids'][choice]
    colorcode = ['red' if x != None else 'blue' for x in flq_map]
    colorcode[0] = 'purple'
    for i,nt in enumerate(node_type_ids):
        if nt == 1:
            colorcode[i] = 'green'

    # PRINTY PRINTY TEST TEST TEST

    # plot graph
    st.write(f"choice = {choice} : {SAMPLE['answers'][choice]}")
    net = Network(notebook=True)
    net.add_nodes(
        range(N),
        value=list([node_scores[x] for x in concept_names]),
        title=concept_names,
        label=concept_names,
        color=colorcode
    )
    for e in edges: net.add_edge(*e)

    net.show('fl_graph.html')
    HtmlFile = open('fl_graph.html', 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=610)
    pass

st.balloons()