# active-phase-mapping

Code for active phase mapping. Paper can be found here: https://arxiv.org/abs/2402.15582

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

## Code notes

The active learning algorithms are in the active_learning directory.
All data generated in the performance testing can be found in the performance_test directory. 
