# Common Sense Question Answering using Hybrid Models for Efficiency and Explainability.
This is the code base for my master thesis (title above)

## Pipeline
_diagram incoming_

## Setup Guide
- git clone --recurse-submodules git@github.com:BrennerG/MA.git
- conda env create -f environment.yml
- conda activate ma
- change line 18 in src/evaluation/eraserbenchmark/rationale_benchmark/metrics.py to "from .utils import"
- read main.py for documentation of parameters

## Running the code
- Set parameters in src/main.py (see main.py for a fully documented example of the parameter space)
- in the future parameters can be manipulated in a .yaml file and a CLI can be used to run the code!

## TODO
* create pipeline with random classifier and full evaluation
  * ~~saving and loading runs~~
  * ~~rnd clf~~
  * ~~eval: competence~~
  * ~~eval: explainability~~
  * ~~eval: efficiency~~
* __BOW__ Baseline
  * bags
  * logistic regression
* __BERT + LIME__ Baseline
  * BERT processing
  * LIME
* __BERT*less__ - create simple GNN Architecture with UD graphs
  * speech to graph
  * GNN
  * attention + explainability!
  * extend to 4Lang
* __QA-GNN Baseline__
  * run QA-GNN
  * run QA-GNN w. UD/4Lang graphs
  * make QA-GNN output compatible with our evaluation suite
* __Minimal__
  * ...
* __Extended__
  * ...
* __Full__
  * ...