# Common Sense Question Answering using Hybrid Models for Efficiency and Explainability.
This is the code base for my master thesis (title above)

## Pipeline
_diagram incoming_

## Setup Guide
- git clone --recurse-submodules git@github.com:BrennerG/MA.git
- conda env create --name ma -f environment.yml
- conda activate ma
- change line 18 in src/evaluation/eraserbenchmark/rationale_benchmark/metrics.py to "from .utils import"
- read main.py for documentation of parameters

## Running the code
- Set parameters in src/main.py (see main.py for a fully documented example of the parameter space)
- in the future parameters can be manipulated in a .yaml file and a CLI can be used to run the code!

## To-Do List
* ~~__create pipeline with random classifier and full evaluation__~~
  - Can we do 'smarter' random? Think majority class only prediction or 'informed' random classifiers _optional_
* ~~__BOW__ Baseline~~
  - ~~this doesn't work~~
* __BERT + LIME__ Baseline
  - experiment w. other methods for aggregating attention?
  - pass data to lime in a better way? other API or sth?
  - experiment w. scaling of lime weights 
* __BERT*less GCN__ - glove + UDA + GCN
  - Implementation:
    - Efficiency metrics?
  - Experiments:
    - experiment w. pooling
    - implement batching?
* __BERT*less__ - glove + UD + GAT
  - ~~parameter sweeps~~
  - one- vs bi-directional edges
  - wh-substitution (fill in the answers for the wh*-word)
  - node weight aggregation (train it?) _important_
  - other graph preprocessing? (ideas in ud_preproc.py)
  - extend to 4Lang
  - apply some methods from QA-GNN
    - QA-GNN actually substitutes the answers in for the 'wh-words'! :O
      - can this work for the other models on ERASER? the wh's might get thrown away
        - the weights tend to be rather high on these words - but there are no guarantees...
        - if the substitution happens as part of preproc, then the model won't know the samples any other way. so we're prb fine
    - QA-GNN adds reverse edges!
* __QA-GNN Baseline__
  - ~~run QA-GNN~~
  - wrap QA-GNN for explainability evaluations _important_
  - swap ConceptNet for 4Lang _optional_
* __Minimal__ = BERT + UD/4Lang + GAT
  - Experiments
    - try BERT*less w. BERT instead of glove
    - node weight aggregation (train it?) _important - do extensively_
    - graph preprocessing? (ideas in ud_preproc.py)
* __Extended__ = __Minimal__ + 4Lang expansions 
  - ...
* __Full__ = __Extended__ + Node/Edge types
  - ...
* __Pipeline__
  - caching (e.g. preprocessed_data, predictions, viz_data, ...)
  - parameter search / dynamic config creation
  - interface with Adam's Graph Classes
  - re-include soft_scores (overlap/plausability metrics back into evaluation)
* __Discussion__ 
- Is evaluating on ERASER without optimizing the model for it actually zero-shot learning?
  - should we train the model to predict the rationales?
  - right now we train for QA and one-shot ERASER
  - should we also try: train for ERASER, one-shot QA?
  - or unify the models, losses, ...

# Environmnts
## exported_
* ma_environment.yml, ma_requirements.txt = main environment
* ma_gnn_environment.yml = environment for QA_GNN experiments
## not exported
* ma4 (<=BERTExperiment)
* eraser_baselines (for eraser BERT+LSTM)
* ma_gnn (development of ud_gcn)
* ma6 (clone of ma_gnn + ma4 packages)