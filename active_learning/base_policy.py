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
from utils import sample_from_posterior


def get_next_y(true_y, design_space, next_x):
    return true_y[:,jnp.newaxis][design_space == next_x]


def compute_distances(dataset, design_space, true_envelope):
    distances = []
    for (i,x) in enumerate(dataset.X): #The parentheses for (i,x) is just to clean it up. dataset.X gives the x-values.
        x_idx = (design_space == x).argmax() #design_space == x returns an array of true and false values.
        #By calling argmax, we're getting the argument where conditions is true.
        distances.append(jnp.abs(dataset.y[i] - true_envelope[1][x_idx])) #Here we're calling the true hull value for a given X. We're calculating the distance away from the hull for the data in our dataset.
    return jnp.array(distances)

def get_next_candidate_baseline(posterior, params, dataset, designs, design_space):
    """
    Baseline active search method based on selecting designs with maximum posterior variance.
    """
    # get covariances and compute log determinant
    covariances = jnp.array([make_preds(dataset, posterior, params, jnp.atleast_2d(x))[1] for x in designs])

    entropy_change = 0.5 * jnp.linalg.slogdet(covariances + 1)[1]
    return designs[entropy_change.argmax()], entropy_change
