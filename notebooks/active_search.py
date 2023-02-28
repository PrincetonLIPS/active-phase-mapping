import jax
import jax.numpy as jnp 
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla
from jax.config import config
config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import grad, jit
import jaxkern as jk
import optax as ox
from jaxutils import Dataset

from gp_model import make_preds, update_model


def get_next_y(true_y, design_space, next_x):
    return true_y[:,jnp.newaxis][design_space == next_x]


def get_next_candidate_baseline(posterior, params, dataset, designs, design_space):
    """
    Baseline active search method based on selecting designs with maximum posterior variance.
    """
    # get covariances and compute log determinant
    covariances = jnp.array([make_preds(dataset, posterior, params, jnp.atleast_2d(x))[1] for x in designs])
    
    entropy_change = 0.5 * jnp.linalg.slogdet(covariances + 1)[1]
    return designs[entropy_change.argmax()], entropy_change


###############################################################################

# Convex-hull-aware active search

###############################################################################

