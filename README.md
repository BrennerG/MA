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
  - ~~saving and loading runs~~
  - ~~rnd clf~~
  - ~~eval: competence~~
  - ~~eval: explainability~~
    - recheck suff implementation
  - ~~eval: efficiency~~
  - __optional__ Can we do 'smarter' random? Think majority class only prediction or 'informed' random classifiers
* ~~__BOW__ Baseline~~
* ~~Change Experiment Structure~~
* __BERT + LIME__ Baseline
  - ~~huggingface clf~~
  - ~~LIME~~
  - ~~Write Evaluation (Competence, Efficiency, Explainability)~~
  - ~~Integrate into Pipeline (write a class with the stuff from bert_cose.py)~~
  - ~~Experimentation / Runs~~
    - ~~aggregate transformer attention for explanations~~ 
      - TODO revisit _optional_
* __BERT*less__ - create simple GNN Architecture with UD graphs
  - ~~text to graph preproc (UD)~~
  - ~~use word embeddings: Glove~~
  - ~~embed into my experiment api~~
  - ~~run on trou~~
  - ~~make it learn~~
  - ~~use GAT~~
  - Experiments
    - parameter sweeps
    - embeddings Glove/BERT/AlBERT/RoBERTa
    - node weight aggregation (train it?) _important - do extensively_
    - graph preprocessing? (ideas in ud_preproc.py)
  - extend to 4Lang
* revisit changes for BERT and Random Baselines in pipeline!
* __QA-GNN Baseline__
  - run QA-GNN
  - wrap QA-GNN for explainability evaluations
  - ~~run QA-GNN w. UD/4Lang graphs~~ _cancelled_
  - ~~make QA-GNN output compatible with my evaluation suite~~ _cancelled_
* __Minimal__ = BERT + UD/4Lang + GAT
  - ...
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
_not all exported_
* ma4 (<=BERTExperiment)
* eraser_baselines (for eraser BERT+LSTM)
* ma_gnn (development of ud_gcn)
* ma6 (clone of ma_gnn + ma4 packages)