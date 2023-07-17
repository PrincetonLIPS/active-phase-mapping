import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla

import gpjax as gpx
from jax import grad, jit
import jaxkern as jk
import optax as ox
from jaxutils import Dataset


def make_preds(dataset, posterior, params, test_x, verbose=False):

    latent_distribution = posterior(params, dataset)(test_x)
    likelihood = gpx.Gaussian(num_datapoints = dataset.n)
    predictive_distribution = likelihood(params, latent_distribution)

    pred_mean = predictive_distribution.mean()
    pred_cov = predictive_distribution.covariance()

    return pred_mean, pred_cov

def update_model(dataset, design_space, rng_key, update_params=False, num_iters=500, lr=1e-3, init_params=None):
    """
    Updates the model or uses default hyperparameters. Assumes zero mean function.
    """

    # Define model
    prior = gpx.Prior(kernel = jk.RBF())
    likelihood = gpx.Gaussian(num_datapoints = dataset.n)
    posterior = prior * likelihood

    if update_params:
        # Update hyperparameters
        mll = jit(posterior.marginal_log_likelihood(dataset, negative=True))
        opt = ox.adam(learning_rate=lr)
        parameter_state = gpx.initialise(posterior, key=rng_key)
        parameter_state = gpx.fit(mll, parameter_state, opt, num_iters=num_iters)
    else:
        # Use default parameters
        if init_params is None:
            parameter_state = gpx.initialise(posterior, key=rng_key,
                                         kernel={"lengthscale": jnp.array([0.2]), "variance": jnp.array([1])},
                                        likelihood={'obs_noise': jnp.array([0.0])})
        else:
            pass # TODO: make it possible to pass in initial parameters for the kernel

    params = parameter_state.params
    pred_mean, pred_cov = make_preds(dataset, posterior, params, design_space)

    return pred_mean, pred_cov, posterior, params
