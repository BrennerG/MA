# MA
Master Thesis (all of it)

# TODO
- [ ] create DataLoader
- [ ] create Pipeline
- [ ] create RandomClassifier
- [ ] create Interface
- [ ] create Logging
...

# Setup Guide
- git clone --recurse-submodules git@github.com:BrennerG/MA.git
- conda env create -f environment.yml
- conda activate ma
- change line 18 in src/evaluation/eraserbenchmark/rationale_benchmark/metrics.py to "from .utils import"
- run main.py
- read XYZ for documentation of parameters