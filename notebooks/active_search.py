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
    def same_tight_1d(y, tight):
        new_hull = convelope(design_space, y).ravel()
        new_tight = y - new_hull < 1e-3
        #points = np.hstack([design_space, y[:, np.newaxis]])
        #hull = ConvexHull(points)
        #new_tight = np.zeros(len(design_space))
        #new_tight[hull.vertices] = 1
        return jnp.all(tight == new_tight)
    
    def same_tight(y_val, tight):
        #new_hull = convelope(design_space, y).ravel()
        #new_hull = is_tight(design_space, y)
        #new_tight = y - new_hull < 1e-3
        #print(y_val)
        new_tight = is_tight(design_space, y_val)[jnp.newaxis, :]
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
    deriv_marg_var = 50 #100
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 200)
    ss = jnp.meshgrid(*[s.ravel()]*D)
    s = jnp.array([sx.flatten() for sx in ss]).T
    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    prod = (design_space @ s.T).T
    
    # compute the conjugate
    lft1 = jnp.max(prod[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    # compute the biconjugate
    lft2 = jnp.max(prod[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal
    
    return lft2



from scipy.spatial import ConvexHull
import numpy as np

def convex_envelope(x, fs):
    ## TODO: make this more efficient
    """Computes convex envelopes of M functions which share a common grid.
    x is an (N, D)-matrix corresponding to the grid in D-dimensional space and fs is an (M, N)-matrix.
    The i-th function is given by the pairs (x[0], fs[i, 0]), ..., (x[N-1], fs[i, N-1]).
    The envelopes are returned as a list of lists.
    The i-th list is of the form [j_1, j_2, ..., j_n] where (X[j_k], fs[i, j_k]) is a point in the envelope.
    
    Keyword arguments:
    x  -- A shape=(N,D) numpy array.
    fs -- A shape=(M,N) or shape=(N,) numpy array."""
    
    #assert(len(fs.shape) <= 2)
    if len(fs.shape) == 1: fs = np.reshape(fs, (-1, fs.shape[0]))
    M, N = fs.shape
    
    #assert(len(x.shape) <= 2)
    #if len(x.shape) == 1: x = np.reshape(x, (-1, 1))
    #assert(x.shape[0] == N)
    D = x.shape[1]
    
    fs_pad = np.empty((M, N+2))
    fs_pad[:, 1:-1], fs_pad[:, (0,-1)] = fs, np.max(fs) + 1.
    
    x_pad = np.empty((N+2, D))
    x_pad[1:-1, :], x_pad[0, :], x_pad[-1, :] = x, x[0, :], x[-1, :]
    
    results = []
    for i in range(M):
        epi = np.column_stack((x_pad, fs_pad[i, :]))
        hull = ConvexHull(epi)
        result = [v-1 for v in hull.vertices if 0 < v <= N]
        #result.sort()
        results.append(np.array(result))
    
    return np.array(results)

def is_vertex(points):
    ### TODO: need to vectorize this function....
    #print(points.shape)
    N, D = points.shape
    vertices = convex_envelope(points[:, :-1], points[:, -1])[0]
    s = np.zeros(N)
    s[vertices] = 1
    return s.astype("bool")
    
def is_tight(design_space, true_y):

    points = jnp.hstack([design_space, true_y[:, jnp.newaxis]])
    _scipy_hull = lambda points: is_vertex(points) 

    result_shape_dtype = jax.ShapeDtypeStruct(
          shape=jnp.broadcast_shapes(true_y.shape),
          dtype='bool')

    return jax.pure_callback(_scipy_hull, result_shape_dtype, points, vectorized=False)