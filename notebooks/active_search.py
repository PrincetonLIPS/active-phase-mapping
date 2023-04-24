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

from elliptical_slice_sampler_jax import elliptical_slice_jax
from gp_model import make_preds, update_model
from search_no_gpjax import sample_from_posterior
from compute_envelope import is_tight, convelope

# non-jax utilities
import numpy as np


def get_next_y(true_y, design_space, next_x):
    return true_y[:,jnp.newaxis][design_space == next_x]

def compute_distances(dataset, design_space, true_envelope):
    distances = []
    for (i,x) in enumerate(dataset.X):
        x_idx = (design_space == x).argmax()
        distances.append(jnp.abs(dataset.y[i] - true_envelope[1][x_idx]))
    return jnp.array(distances)

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
# No-qhull version

###############################################################################

def ess_and_estimate_entropy_gpjax(putative_x, design_space, dataset, posterior, params, s, y, cK, rng_key, J=50):
    """
    Get samples of function conditioned on tights, get samples of y preds conditioned on 
        these samples, and then estimate the entropy.
    """
    # sample J*3 number of points but only keep the last J 
    def same_tight(y, tight):
        new_hull = convelope(design_space, y).ravel()
        new_tight = y - new_hull < 1e-10 #1e-3
        return jnp.all(tight == new_tight)

    # samples of f given tights
    totsamps = J*3
    samps = elliptical_slice_jax(y.ravel(), lambda x: jnp.log(same_tight(x, s)), cK, totsamps, rng_key)
    test_samps = samps[totsamps-J:totsamps]
    
    x_ind = (putative_x == design_space).argmax()
    ystars = test_samps[:, x_ind]
    
    # compute a KDE estimator of density p(y | s, data, putative_x)
    ypred_kde = jsps.gaussian_kde(ystars, bw_method='scott', weights=None)
    
    # evaluate the log probability on the samples y^{(j)}
    return -ypred_kde.logpdf(ystars).mean() # inner MC estimate

def compute_IG_putative_x_noqhull(putative_x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T=100, J=200):
    """
    Compute a Monte Carlo approximation of the IG w.r.t. T samples of s_t ~ p(s | data).
    
    The inner entropy is approximated via Monte Carlo + a KDE estimator constructed from the samples. 
    (TODO: don't use the same data twice)
    
    T: number of samples for the outer expectation
    J: number of ESS samples (controls the # of samples for the inner MC too?)
    """

    def entropy_est_wrap(args):
        tights_i, pred_Y_i = args
        return ess_and_estimate_entropy_gpjax(putative_x, design_space, dataset, posterior, params, tights_i, pred_Y_i, pred_cK, rng_key, J=J)
    #ess_and_estimate_entropy_gpjax(putative_x, design_space, dataset, posterior, params, s, y, cK, rng_key, J=50)
    
    ventropy_est = jax.jit(jax.vmap(entropy_est_wrap, in_axes=((1,1),)))
    #ventropy_est = jax.vmap(entropy_est_wrap, in_axes=((1,1),))

    entropies = ventropy_est((tights, pred_Y))  
    ##entropies = jnp.array([ess_and_estimate_entropy_gpjax_test(putative_x, design_space, dataset, posterior, params, tights[:,i], pred_Y[:,i], 
    ##                                                           pred_cK, rng_key, J=J) for i in range(len(tights))])
    
    # estimate of the second term in the EIG
    return entropies.mean()

def get_next_candidate_noqhull(posterior, params, dataset, designs, design_space, rng_key, T=30, J=40, tol=1e-3):
    """
    Given current data and a list of designs, computes an IG score for each design. 
    
    T: number of outer MC samples
    J: number of inner MC samples
    tol: tolerance for considering what is tight w.r.t. the convex hull
    
    Returns the best design and the list of scores. 
    """

    # updates the model and samples T functions and computes their envelopes. here we evaluate functions only at points in the design space
    #pred_mean, pred_cov, pred_cK, pred_Y, envelopes = update_model(data, T, design_space) 
    # compute the vector of indicators
    pred_mean, pred_cov = make_preds(dataset, posterior, params, design_space)
    pred_Y, envelopes, pred_cK = sample_from_posterior(pred_mean, pred_cov, design_space, T, get_env=True)

    tights = jnp.abs(envelopes.T - pred_Y) < tol ## NOTE: we transposed the shape from what it was before
    
    # TODO: move the lambda function into the vmap to make it cleaner
    compute_IG_putative_wrap = lambda x: compute_IG_putative_x_noqhull(x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T = T, J = J) 
    compute_IG_vmap = jax.jit(jax.vmap(compute_IG_putative_wrap, in_axes=0))
    
    # TODO: faster to find relevant indicies?
    _, pred_cov_designs = make_preds(dataset, posterior, params, designs)
    #curr_entropy = jnp.log(jnp.sqrt(2 * jnp.pi * jnp.e * jnp.diag(pred_cov_designs)))
    curr_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.diag(pred_cov_designs)) 
    mean_entropy = compute_IG_vmap(designs)
    entropy_change = curr_entropy - mean_entropy
    
    return designs[entropy_change.argmax()], entropy_change

###############################################################################

# Convex-hull-aware active search
# Qhull version

###############################################################################

def ess_and_estimate_entropy_gpjax_test(putative_x, design_space, dataset, posterior, params, s, y, cK, rng_key, J=50):
    """
    Get samples of function conditioned on tights, get samples of y preds conditioned on 
        these samples, and then estimate the entropy.
    """
    # sample J*3 number of points but only keep the last J 
    
    def same_tight_1d(y, tight):
        new_hull = convelope(design_space, y).ravel()
        new_tight = y - new_hull < 1e-3
        return jnp.all(tight == new_tight)

    def same_tight(y, tight):
        new_tight = is_tight(design_space, y)
        return jnp.all(tight == new_tight)

    # samples of f given tights
    totsamps = J*3
    samps = elliptical_slice_jax(y.ravel(), lambda x: jnp.log(same_tight(x, s)), cK, totsamps, rng_key)
    test_samps = samps[totsamps-J:totsamps]
    
    x_ind = (putative_x == design_space).argmax()
    ystars = test_samps[:, x_ind]
    
    # compute a KDE estimator of density p(y | s, data, putative_x)
    ypred_kde = jsps.gaussian_kde(ystars, bw_method='scott', weights=None)
    
    # evaluate the log probability on the samples y^{(j)}
    return -ypred_kde.logpdf(ystars).mean() # inner MC estimate

def compute_IG_putative_x_gpjax(putative_x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T=100, J=200):
    """
    Compute a Monte Carlo approximation of the IG w.r.t. T samples of s_t ~ p(s | data).
    
    The inner entropy is approximated via Monte Carlo + a KDE estimator constructed from the samples. 
    (TODO: don't use the same data twice)
    
    T: number of samples for the outer expectation
    J: number of ESS samples (controls the # of samples for the inner MC too?)
    """

    def entropy_est_wrap(args):
        tights_i, pred_Y_i = args
        return ess_and_estimate_entropy_gpjax_test(putative_x, design_space, dataset, posterior, params, tights_i, pred_Y_i, pred_cK, rng_key, J=J)
    
    ventropy_est = jax.jit(jax.vmap(entropy_est_wrap, in_axes=((1,1),)))
    entropies = ventropy_est((tights, pred_Y))  
    
    # estimate of the second term in the EIG
    return entropies.mean()
    

def get_next_candidate_qhull(posterior, params, dataset, designs, design_space, rng_key, T=30, J=40, tol=1e-3):
    """
    Given current data and a list of designs, computes an IG score for each design. 
    
    T: number of outer MC samples
    J: number of inner MC samples
    tol: tolerance for considering what is tight w.r.t. the convex hull
    
    Returns the best design and the list of scores. 
    """

    # updates the model and samples T functions and computes their envelopes. here we evaluate functions only at points in the design space 
    pred_mean, pred_cov = make_preds(dataset, posterior, params, design_space)
    pred_Y, _, pred_cK = sample_from_posterior(pred_mean, pred_cov, design_space, T, get_env=False)
        
    #tights = jnp.abs(envelopes.T - pred_Y) < tol ## NOTE: we transposed the shape from what it was before
    get_tights = jax.jit(jax.vmap(lambda y: is_tight(design_space, y), in_axes=(1,)))
    tights = get_tights(pred_Y).T
    
    
    # TODO: move the lambda function into the vmap to make it cleaner
    compute_IG_putative_wrap = lambda x: compute_IG_putative_x_gpjax(x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T = T, J = J) 
    compute_IG_vmap = jax.jit(jax.vmap(compute_IG_putative_wrap, in_axes=0))
    
    
    # TODO: faster to find relevant indicies?
    _, pred_cov_designs = make_preds(dataset, posterior, params, designs)
    curr_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.diag(pred_cov_designs)) 
    mean_entropy = compute_IG_vmap(designs)
    entropy_change = curr_entropy - mean_entropy
    return designs[entropy_change.argmax()], entropy_change


