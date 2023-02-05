import json
import streamlit as st
import numpy as np


# order of data: graph_veces, sent_vecs, z_vecs
PREDS_PATH = 'notebooks/data/comp_vs_suff/PREDS_DATA.jsonl'
COMP_PATH = 'notebooks/data/comp_vs_suff/COMP_DATA.jsonl'
SUFF_PATH = 'notebooks/data/comp_vs_suff/SUFF_DATA.jsonl'
BATCH_SIZE = 16
DIMS = {'sent_vecs':1024, 'graph_vecs':100, 'z_vecs':100}

#@st.cache()
def load(path):
    data = []

    # READ DATA
    with open(path, 'r') as file:
        for i,line in enumerate(file):
            sample = json.loads(line)
            for k in sample: sample[k] = np.array(sample[k])
            data.append(sample)
    
    # DATA TRANSPOSE
    keys = data[0].keys()
    transposed_data = {k:[] for k in keys}
    for x in data:
        for k in keys:
            transposed_data[k].append(x[k])
    
    # DATA DEBATCH
    debatched_data = {k:[] for k in keys}
    for k in keys:
        for i,x in enumerate(transposed_data[k]):
            reshaped_x = x.reshape((-1,5,DIMS[k]))
            for s in reshaped_x:
                debatched_data[k].append(s)

    return debatched_data

def difference(data1, data2):
    keys = data1.keys()
    diff = {k:[] for k in keys}
    for k in keys:
        for x,y in zip(data1[k], data2[k]):
            diff[k].append(abs((x-y).mean()))
    return {k:np.array(v).mean() for k,v in diff.items()}

with st.container():
    pred_data = load(PREDS_PATH)
    comp_data = load(COMP_PATH)
    suff_data = load(SUFF_PATH)

    st.write(
        difference(pred_data, pred_data),
        difference(pred_data, comp_data),
        difference(pred_data, suff_data),
        difference(comp_data, suff_data),
    )

    og_statements = [json.loads(line) for line in open('data/qa_gnn/dev.statement.jsonl')]
    comp_statements = [json.loads(line) for line in open('notebooks/data/comp_vs_suff/comp.dev.statement.jsonl')]
    suff_statements = [json.loads(line) for line in open('notebooks/data/comp_vs_suff/suff.dev.statement.jsonl')]

    i = 201
    st.write(
        og_statements[i],
        comp_statements[i],
        suff_statements[i]
    )