import json

import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

from pyvis.network import Network
import streamlit.components.v1 as components


# PARAMS
EXPERIMENT = "bright-kumquat-28"
MA_FILE = f"data/experiments/{EXPERIMENT}/viz_data.json"
# GENERAL
RND_SAMPLE_FILE = f"notebooks/data/general/random_quali_select_from_test.json"
STATEMENT_FILE = f"notebooks/data/general/test.statement.jsonl"
# QAGNN DATA
CONCEPT_FILE = f"notebooks/data/og_qagnn/concept.txt"
OG_DATA_TRAIN = f"data/csqa_raw/train_rand_split.jsonl"
OG_DATA_DEV = f"data/csqa_raw/dev_rand_split.jsonl"
OG_DATA_TEST= f"data/csqa_raw/test_rand_split_no_answers.jsonl"
OG_FILE = f"notebooks/data/og_qagnn/SAVED_DIC.json"
# SCRIPT CONSTANTS
NODE_COLORS = ['red', 'green', 'blue', 'purple']

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
    sample_dic = {k:v for k,v in dict(zip(rnd_sample_idx, rnd_sample_ids)).items()}
    sample_dic = {key:sample_dic[key] for key in sorted(sample_dic.keys())}
    rnd_sample_idx = sample_dic.keys()
    rnd_sample_ids = sample_dic.values()

    # separate concepts and raw_data
    id2concept = raw_data.pop('4L_id2concept')
    data = {k:[] for k in raw_data.keys()}
    for k,v in raw_data.items():
        if v != None:
            for i,x in enumerate(v):
                if i in rnd_sample_idx:
                    data[k].append(x)
        else:
            data[k] = [None] * len(rnd_sample_idx)
    
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
    
    return list(sample_dic.values()), data, id2concept

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
    statement_data = []
    with open(STATEMENT_FILE, 'r') as file:
        for line in file:
            statement_data.append(json.loads(line))

    raw_data['statement_data'] = statement_data
    rnd_sample_idx = [x['id'] for x in raw_data['statement_data']]
    rnd_sample_idx = [rnd_sample_idx.index(x) for x in rnd_sample_ids]

    raw_data.pop('id2concept')
    #st.write(raw_data.keys())
    data = {k:[x for i,x in enumerate(v) if i in rnd_sample_idx] for k,v in raw_data.items()}

    # load id2concept
    with open(CONCEPT_FILE) as _file_:
        id2concept = [line for line in _file_]

    return data, id2concept

def create_og_graph(data, option_id, id2concept, value="node_relevance"):
    C = np.argmax(data['logits'][option_id]) # choice
    #st.write(f"choice = {C} : {data['statement_data'][option_id]['question']['choices'][C]['text']}")

    # nodes
    concept_ids = data['concept_ids'][option_id][C]
    concept_names = [id2concept[x-1] if x!=0 else id2concept[0] for x in concept_ids]
    node_type_ids = data['node_type_ids'][option_id][C]
    if value=='node_relevance':
        node_scores = [x[0] for x in data['node_scores'][option_id][C]]
    elif value=='attn':
        node_scores = data['attn'][option_id][C]
    colorcode = [NODE_COLORS[x] for x in node_type_ids]
    N = len(concept_ids)

    # edges
    raw_edges = data['edge_index_orig'][option_id][C]
    edges = [(raw_edges[0][i], raw_edges[1][i]) for i in range(len(raw_edges[0]))]
    E = len(edges)

    # construct graph
    #st.write(f"N={N}, E={E}")
    G = nx.Graph()
    G.add_nodes_from(
        [(i,{'value':val, 'title':name, 'label':name, 'color':color}) for i, (val, name, color) in enumerate(zip(node_scores, concept_names, colorcode))]
    )
    G.add_edges_from(edges)
    return G

def create_4L_graph_alt(sample, id2concept, value='attention'):
    C = np.argmax(sample['logits']) # choice

    # nodes
    concept_ids = sample['concept_ids'][C]
    concept_names = [id2concept[str(x)].strip() for x in concept_ids]
    node_type_ids = sample['node_type_ids'][C]
    if value=='node_relevance':
        node_scores = sample['node_scores'][C]
    elif value=='attn':
        _attn = [x[C] for x in sample['attn']]
        node_scores = np.array(_attn).mean(axis=0)
    colorcode = [NODE_COLORS[x] for x in node_type_ids]
    N = len(concept_ids)

    # edges
    raw_edges = sample['edge_index_orig'][C]
    edges = [(raw_edges[0][i], raw_edges[1][i]) for i in range(len(raw_edges[0]))]
    E = len(edges)

    G = nx.Graph()
    G.add_nodes_from(
        [(i,{'value':val, 'title':name, 'label':name, 'color':color}) for i, (val, name, color) in enumerate(zip(node_scores, concept_names, colorcode))]
    )
    G.add_edges_from(edges)
    return G

def create_4L_graph(sample, value='node_relevance'):

    # choice
    logits = sample['logits']
    choice = np.argmax(logits)
    #st.write(f"choice = {choice} : { sample['answers'][choice]}")

    # nodes
    concept_names = sample['4L_concept_names'][choice]
    N = len(concept_names)
    if value=='node_relevance':
        node_scores = sample['node_scores'][choice]
        if 'Z_VEC' in node_scores: node_scores.pop('Z_VEC')
        node_scores = list(node_scores.values()) # right order?
    elif value=='attn':
        node_scores = np.array(sample['attn']).mean(axis=0)[choice][:N]

    # edges
    edges = sample['4L_edges'][choice][0]
    edge_types = sample['4L_edges'][choice][1]
    E = len(edges)

    # colors
    flq_map = SAMPLE['4L_map'][choice]
    colorcode = ['red' if x != None else 'blue' for x in flq_map]

    # construct graph
    #st.write(f"N={N}, E={E}")
    G = nx.Graph()
    G.add_nodes_from(
        [(i,{'value':val, 'title':name, 'label':name, 'color':color}) for i, (val, name, color) in enumerate(zip(node_scores, concept_names, colorcode))]
    )
    G.add_edges_from(edges)
    return G

def show_graph(G, jiggle=True):
    graph_path = 'notebooks/tmp/fl_graph.html'

    net = Network(notebook=True)
    net.from_nx(G)
    net.toggle_physics(jiggle)
    net.show(graph_path)

    HtmlFile = open(graph_path, 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=610)

def filter_for_special_nodes(G, hops=0):
    selection = []
    for n,ndic in G.nodes(data=True):
        if ndic['color'] != 'blue':
            selection.append(n)
    
    for h in range(hops):
        extension = []
        for s in selection:
            extension.extend(list(G.neighbors(s)))
        selection.extend(extension)

    def node_filter(n):
        return n in selection
    
    return nx.subgraph_view(G, node_filter)

def filter_by_value(G, n=200):
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]['value'], reverse=True)[:n]

    def node_filter(n):
        return n in sorted_nodes

    return nx.subgraph_view(G, node_filter)

def get_og_sample(sid):
    all_raw = []
    with open(OG_DATA_TRAIN, 'r') as file:
        for line in file:
            all_raw.append(json.loads(line))
    with open(OG_DATA_DEV, 'r') as file:
        for line in file:
            all_raw.append(json.loads(line))
    with open(OG_DATA_TEST, 'r') as file:
        for line in file:
            all_raw.append(json.loads(line))
    for sample in all_raw:
        if sample['id'] == sid:
            return sample
    st.write(f"SAMPLE {sid} NOT FOUND! :(")

def draw_attn_bar_plot(_map, attn, question_tokens):
    res = {n:a for n,a in zip(_map, attn) if n != None}
    viz = [(x,res[i]) if i in res else (x,0) for i,x in enumerate(question_tokens)]
    df = pd.DataFrame(viz, columns=['tokens', 'attn'])
    fig = px.bar(df,
                title="...",
                x='tokens', 
                y='attn')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    return True

def get_viz_data_for_og(sample, id2concept):
    C = np.argmax(sample['logits'])
    _attn = np.array(sample['attn']).reshape(5,2,200)
    attn = _attn[C].mean(axis=0)
    concept_ids = sample['concept_ids'][C]
    concept_names = [id2concept[x-1].strip() for x in concept_ids]
    question_tokens = sample['statement_data']['question']['stem'].split()
    og_map = [question_tokens.index(x) if x in question_tokens else None for x in concept_names]
    res = {n:a for n,a in zip(og_map, attn) if n != None}
    viz = [(x,res[i]) if i in res else (x,0) for i,x in enumerate(question_tokens)]
    return viz

def get_viz_data_for_fl(sample):
    C = np.argmax(SAMPLE['logits'])
    _attn = np.array([x[C] for x in SAMPLE['attn']])
    attn = np.mean(_attn, axis=0)
    concepts = SAMPLE['4L_concept_names'][C]
    fl_map = SAMPLE['4L_map'][C]
    question_tokens = SAMPLE['question'].split()
    res = {n:a for n,a in zip(fl_map, attn) if n != None}
    viz = [(x,res[i]) if i in res else (x,0) for i,x in enumerate(question_tokens)]
    return viz

################## ################## ################## ##################
################## ############### MAIN ################ ##################
################## ################# ################### ##################

st.set_page_config(layout="wide")
sel_ids, data, fl_id2concept = load_ma_data(MA_FILE)

################## ################## SIDEBAR ################## ##################

with st.sidebar:
    option = st.selectbox("which sample?", sel_ids)
    option_id = sel_ids.index(option)
    st.write(option_id)

################## ################## HEADER ################## ##################

with st.container():
    #st.write(pd.DataFrame(data))
    SAMPLE = {k:v[option_id] for k,v in data.items()}
    OG_SAMPLE = get_og_sample(option)
    st.title(OG_SAMPLE['question']['stem']) # TODO use real question here
    st.write({k:v for k,v in SAMPLE.items() if k in ['question', 'answers', 'label']})

################## ################## COLS ################## ##################

LEFT_C, RIGHT_C = st.columns(2, gap='medium')

with LEFT_C:
    st.subheader("cpnet + QA-GNN")
    data, og_id2concept = load_og_data(OG_FILE, RND_SAMPLE_FILE)
    hops = st.slider('hops?', 0, 5, 0, key='lhops')
    cutoff = st.slider('cutoff?', 0, 200, 200, key='lcutoff')
    value_shown = st.selectbox("node_values? attn / node_relevance", ['attn', 'node_relevance'], key='lvalue')

    net = create_og_graph(data, option_id, og_id2concept, value=value_shown)
    net = filter_for_special_nodes(net, hops=hops)
    net = filter_by_value(net, n=cutoff)
    st.write(len(net.nodes), len(net.edges))
    show_graph(net)

    LSAMPLE = {k:v[option_id] for k,v in data.items()}

################## ################## SEPARATOR ################## ##################

with RIGHT_C:
    st.subheader("4Lang + QA-GNN")
    hops = st.slider('hops?', 0, 5, 0, key='rhops')
    cutoff = st.slider('cutoff?', 0, 200, 200, key='rcutoff')
    value_shown = st.selectbox("node_values? attn / node_relevance", ['attn', 'node_relevance'], key='rvalue')

    net = create_4L_graph_alt(SAMPLE, fl_id2concept, value=value_shown)
    net = filter_for_special_nodes(net, hops=hops)
    net = filter_by_value(net, n=cutoff)
    st.write(len(net.nodes), len(net.edges))
    show_graph(net)

################## ################## SEPARATOR ################## ##################

with st.container():
    st.subheader("Attention")
    og_viz = get_viz_data_for_og(LSAMPLE, og_id2concept)
    fl_viz = get_viz_data_for_fl(SAMPLE)
    og_viz_dict = {k:v for (k,v) in og_viz}
    _viz = [(k,v,og_viz_dict[k]) if k in og_viz_dict else (k,v,0) for k,v in fl_viz]
    viz = []
    for (token, og, fl) in _viz:
        viz.append((token,og,'w. cpnet'))
        viz.append((token,fl,'w. 4Lang'))
    df = pd.DataFrame(
        viz, columns=['token', 'attn', 'type']
    )
    fig = px.bar(
        df,
        x='token',
        y='attn',
        color='type',
        barmode='group'
    )
    fig.update_xaxes(tickangle=315)
    st.plotly_chart(fig, use_container_width=True)

st.balloons()