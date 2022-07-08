# Common Sense Question Answering using Hybrid Models for Efficiency and Explainability.
This is the code base for my master thesis with the name above.

## Setup Guide
- git clone --recurse-submodules git@github.com:BrennerG/MA.git
- conda env create -f environment.yml
- conda activate ma
- change line 18 in src/evaluation/eraserbenchmark/rationale_benchmark/metrics.py to "from .utils import"
- run main.py
- read XYZ for documentation of parameters

## Running the code
- Set parameters in src/main.py (see main.py for a fully documented example of the parameter space)
- in the future parameters can be manipulated in a .yaml file and a CLI can be used to run the code!