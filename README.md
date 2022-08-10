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

## TODO
* ~~__create pipeline with random classifier and full evaluation__~~
  - ~~saving and loading runs~~
  - ~~rnd clf~~
  - ~~eval: competence~~
  - ~~eval: explainability~~
    - recheck suff implementation
  - ~~eval: efficiency~~
* ~~__BOW__ Baseline~~
* __BERT + LIME__ Baseline
  - ~~huggingface clf~~
  - ~~LIME~~
  - ~~Write Evaluation (Competence, Efficiency, Explainability)~~
  - ~~Integrate into Pipeline (write a class with the stuff from bert_cose.py)~~
  - ~~cleanup TODOs~~
  - Experimentation / Runs
    - Preparation
      - ~~rework debug_flag~~
      - ~~skip_evaluation parameter~~
      - ~~cuda yes/no flag~~
      - ~~FIX: new pipeline doesnt learn, but old did (confirmed) SOLUTION: remove debug flag and set LIMIT manually in huggingface everytime!~~
      - ~~Error on troubadix: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! loss plots~~
      - why is new pipeline slower than the old pipeline? _optional_
    - high epoch run
      - run
      - eval
      - viz
    - full LIME evaluation run (high #features and #permutations)
      - run
      - eval
      - viz
    - aggregate transformer attention for explanations _optional_
      - run
      - eval
      - viz
* ~~Change Experiment Structure~~
* __BERT*less__ - create simple GNN Architecture with UD graphs
  - text to graph preproc (UD)
  - use word embeddings: Glove
  - GNN / GAT
  - how to aggregate attention?
    - do extensive experimentation here! (might be a crucial point of discussion)
  - extend to 4Lang
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