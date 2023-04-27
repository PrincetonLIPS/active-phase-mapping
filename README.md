# active-phase-mapping

Code for active phase mapping.

## Setup notes

Currently uses jax and
[gpjax](https://gpjax.readthedocs.io/en/latest/installation.html), which relies
on an updated version of python:

```
conda create --name apm python=3.10
conda activate apm
conda install -y scipy
pip install jax jaxlib
pip install gpjax
```

If using GPUs with jax,
```
pip install --upgrade "jax[cuda]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
```

Other package dependencies:
* matplotlib
* python-ternary
* pandas



## Code notes

Currently active_search.py contains the active search algorithms and
relevant plotting utilities, and gp_model.py contains the GP-related model
utilities.


TODO:
* Notebook: 1-simplex synthetic example in a notebook
* Notebook: 2-simplex synthetic example in a notebook
* Notebook: 2-simplex on alloy data with hyperparameter learning
