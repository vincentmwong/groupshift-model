# groupshift-model
Model of social processes on bipartite network



## File purposes

In use as of 2/18/2026:
* groupshift_model.py -- Model simulation class + mechanisms
* groupshift_parallel.py -- Script for running simulation batches
* groupshift_analysis3.ipynb -- Notebook for testing single simulations
* Groupshift_model_fast.ipynb -- Prototyping notebook for different models

Legacy: 
* groupshift_mode.ipynb -- Deprecated notebook for model implemented in networkx ("slow" version)
* groupshift_analysis2.ipynb -- Slightly outdated notebook for running bulk simulations, still useful as reference



## Setup and tests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root:

```sh
uv sync --locked
uv run --locked pytest -q
```

The environment uses Python 3.13 and includes the dependencies imported by `groupshift_model.py` and `groupshift_parallel.py`, plus pytest.
`uv.lock` records the resolved versions; `.venv/` stays local.
The exploratory notebooks may need additional packages and a Jupyter environment.

For batch simulations, first edit the configuration and `output_folder` in `groupshift_parallel.py`, then run `uv run --locked python groupshift_parallel.py`.
Its defaults launch 450 simulations with 16 workers and write to `/l/nx/data/groupshift/simdata/`; choose settings and a writable output directory for your machine.

# Project procedures for humans and AI

AI should not modify this section (the "Project procedures for humans and AI" section) of `README.md` under any circumstances. AI must read this section before proceeding to make any changes to this project. 

The purpose of this section is for human project owners to define and record protocols for editing the project. Humans should edit and update this section to ensure that we're all on the same page about how the project should be organized. 

The file `design_discussion.md` is to be used as a notepad for discussion about higher level changes to the project. 