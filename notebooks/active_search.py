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

###############################################################################

def get_next_candidate(posterior, params, dataset, designs, design_space, rng_key, T=30, J=40, tol=1e-3):
    """
    Given current data and a list of designs, computes an IG score for each design. 
    
    T: number of outer MC samples
    J: number of inner MC samples
    tol: tolerance for considering what is tight w.r.t. the convex hull
    
    Returns the best design and the list of scores. 
    """

    # updates the model and samples T functions and computes their envelopes. here we evaluate functions only at points in the design space
    pred_mean, pred_cov = make_preds(dataset, posterior, params, design_space)
    pred_Y, envelopes, pred_cK = sample_from_posterior(pred_mean, pred_cov, design_space, T)
    # compute the vector of indicators
    tights = jnp.abs(envelopes.T - pred_Y) < tol 
    
    # TODO: move the lambda function into the vmap to make it cleaner
    compute_IG_putative_wrap = lambda x: compute_IG_putative_x(x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T = T, J = J) 
    compute_IG_vmap = jax.jit(jax.vmap(compute_IG_putative_wrap, in_axes=0))
    
    # TODO: faster to find relevant indicies?
    _, pred_cov_designs = make_preds(dataset, posterior, params, designs)
    curr_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.diag(pred_cov_designs)) 
    mean_entropy = compute_IG_vmap(designs)
    
    entropy_change = curr_entropy - mean_entropy
    return designs[entropy_change.argmax()], entropy_change


def compute_IG_putative_x(putative_x, design_space, dataset, posterior, params, pred_cK, pred_Y, tights, rng_key, T=100, J=200):
    """
    Compute a Monte Carlo approximation of the IG w.r.t. T samples of s_t ~ p(s | data).
    
    The inner entropy is approximated via Monte Carlo + a KDE estimator constructed from the samples. (TODO: don't use the same data twice)
    
    T: number of samples for the outer expectation
    J: number of ESS samples (controls the # of samples for the inner MC too?)
    """

    def entropy_est_wrap(args):
        tights_i, pred_Y_i = args
        return ess_and_estimate_entropy(putative_x, design_space, dataset, posterior, params, tights_i, pred_Y_i, pred_cK, rng_key, J=J)
    ventropy_est = jax.jit(jax.vmap(entropy_est_wrap, in_axes=((1,1),)))
    entropies = ventropy_est((tights, pred_Y))  
    
    # estimate of the second term in the EIG
    return entropies.mean()
    
def ess_and_estimate_entropy(putative_x, design_space, dataset, posterior, params, s, y, cK, rng_key, J=50):
    """
    Get samples of function conditioned on tights, get samples of y preds conditioned on 
        these samples, and then estimate the entropy.
    """
    # sample J*3 number of points but only keep the last J 
    def same_tight(y, tight):
        #new_hull = convelope(design_space, y).ravel()
        #new_tight = y - new_hull < 1e-3
        points = np.hstack([design_space, y[:, np.newaxis]])
        hull = ConvexHull(points)
        new_tight = np.zeros(len(design_space))
        new_tight[hull.vertices] = 1
        return jnp.all(tight == new_tight)

    # samples of f given tights
    totsamps = J*3
    samps = elliptical_slice_jax(y.ravel(), lambda x: jnp.log(same_tight(x, s)), cK, totsamps, rng_key)
    test_samps = samps[totsamps-J:totsamps]
    
    x_ind = (putative_x == design_space).argmax()
    ystars = test_samps[:, x_ind]
    
    """
    # get 1d predictive y samples at values of the full design space
    def make_pred_single(train_y):
        data_values = Dataset(X = design_space, y = train_y[:, jnp.newaxis])
        pred_mean, pred_cov = make_preds(data_values, posterior, params, jnp.atleast_2d(putative_x))
        return pred_mean.ravel()[0], pred_cov.ravel()[0]
    makepred_vmap = jax.jit(jax.vmap(make_pred_single, in_axes=(0,)))

    # get predictive means and variances and samples according to these parameters
    mus, sigmas = makepred_vmap(test_samps)
    ystars = jrnd.multivariate_normal(rng_key, mus, jnp.eye(len(mus))*sigmas) # TODO: just rescale by cholesky + mean
    """
    
    # compute a KDE estimator of density p(y | s, data, putative_x)
    ypred_kde = jsps.gaussian_kde(ystars, bw_method='scott', weights=None)
    
    # evaluate the log probability on the samples y^{(j)}
    return -ypred_kde.logpdf(ystars).mean() # inner MC estimate
    

# TODO: check this for correctness
def convelope(design_space, knot_y):

    N, D = design_space.shape
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO: 
    #deriv_marg_var = np.max(jnp.diag(d_kernel(knot_x, knot_x, ls)))
    deriv_marg_var = 100
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 500)
    ss = jnp.meshgrid(*[s.ravel()]*D)
    s = jnp.array([sx.flatten() for sx in ss]).T
    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    prod = (design_space @ s.T).T
    
    # compute the conjugate
    lft1 = jnp.max(prod[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    # compute the biconjugate
    lft2 = jnp.max(prod[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal
    
    return lft2



