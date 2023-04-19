import jax
import jax.numpy as jnp 
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla
from jax.config import config
config.update("jax_enable_x64", True)

import numpy.random as npr 

ls_default = 0.1

# Non GPjax code
def kernel_rbf(x1, x2):
    """
    Squared expoential kernel with lengthscale ls. 
    """
    ls=ls_default; v=1
    #ls = params["lengthscale"]; v = params["variance"]
    return v * jnp.exp(-0.5 * jnp.linalg.norm(x1-x2) ** 2 / ls ** 2)

def K(kernel, xs, ys):
    """
    Compute a Gram matrix or vector from a kernel and an array of data points.
    Args:
    kernel: callable, maps pairs of data points to scalars.
    xs: array of data points, stacked along the leading dimension.
    Returns: A 2d array `a` such that `a[i, j] = kernel(xs[i], xs[j])`.
    """
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(ys))(xs)


def make_preds(train_x, train_y, test_x):
    """
    Given data, get pred_mean and pred_cov. 
    
    TODO: change argument to data instead of train_x, train_y?
    """
    train_K = K(kernel_rbf, train_x, train_x) + 1e-6* jnp.eye(train_x.shape[0])
    cross_K = K(kernel_rbf, train_x, test_x)
    kappa_K = K(kernel_rbf, test_x, test_x)

    # Predictive parameters.
    train_cK = spla.cholesky(train_K)
    cross_solve = spla.cho_solve((train_cK,  False), cross_K)
    pred_mean = train_y.T @ cross_solve
    pred_cov  = kappa_K - cross_K.T @ cross_solve + 1e-6*jnp.eye(test_x.shape[0])
    
    return pred_mean, pred_cov

# Compute the convex envelope given x and y values
def convelope_old(knot_x, knot_y):
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO: 
    deriv_marg_var = jnp.max(jnp.diag(d_kernel(knot_x, knot_x, ls)))
    deriv_marg_var = 100
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 500)

    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    print(knot_y.shape)
    sx = s[:,jnp.newaxis] * knot_x[jnp.newaxis,:] # num_dual x num_primal
    print(sx.shape)
    # compute the conjugate
    lft1 =jnp.max(sx[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    print(lft1.shape)
    # compute the biconjugate
    lft2 = jnp.max(sx[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal
    return lft2

def get_next_candidate_baseline(data, designs, design_space):
    pred_mean, pred_cov, pred_cK, pred_Y, envelopes = update_model(data, 20, design_space)
    
    train_x, train_y = data
    
    # get covariances and compute log determinant
    covariances = jnp.array([make_preds(train_x, train_y, x)[1] for x in designs])
    entropy_change = 0.5 * jnp.linalg.slogdet(covariances + 1)[1]
    return entropy_change.argmax(), entropy_change

def update_model(data, T, design_space):
    N_designs = design_space.shape[0]
    train_x, train_y = data
    
    # TODO: add function to update model hyperparameters
    params = 1
    
    # sample functions ~ posterior
    pred_mean, pred_cov = make_preds(train_x, train_y, design_space) ### ? 
    pred_cK = spla.cholesky(pred_cov)
    # get T samples from the posterior
    pred_Y = pred_cK.T @ npr.randn(N_designs, T) + pred_mean[:,jnp.newaxis]
    # get s by computing the vector of tights w.r.t. posterior samples
    envelopes = convelope(design_space, pred_Y.T)
    
    return pred_mean, pred_cov, pred_cK, pred_Y, envelopes

# not needed anymore
def add_observation(data, true_y, next_x, design_space, design_inds):
    train_x, train_y = data
    #train_x = jnp.concatenate([train_x, jnp.array([design_space[design_inds[next_x]]])])
    #train_y = jnp.concatenate([train_y, jnp.array([true_y[design_inds[next_x]]])])
    train_x = jnp.concatenate([train_x, jnp.array([design_space[next_x]])])
    train_y = jnp.concatenate([train_y, jnp.array([true_y[next_x]])])
    return (train_x, train_y)


def sample_from_posterior(pred_mean, pred_cov, design_space, T, get_env=False):
    # TODO: clean this up -- not necessary
    # sample functions ~ posterior
    N_designs = design_space.shape[0]
    pred_cK = spla.cholesky(pred_cov)
    # get T samples from the posterior
    pred_Y = pred_cK.T @ npr.randn(N_designs, T) + pred_mean[:, jnp.newaxis]
    # get s by computing the vector of tights w.r.t. posterior samples
    if get_env:
        envelopes = convelope(design_space, pred_Y.T)
    else:
        envelopes = None
    
    return pred_Y, envelopes, pred_cK


def convelope(design_space, knot_y):

    N, D = design_space.shape
    deriv_marg_var = 100
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO: 
    #deriv_marg_var = np.max(jnp.diag(d_kernel(knot_x, knot_x, ls)))
    #print(deriv_marg_var)
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 500)
    ss = jnp.meshgrid(*[s.ravel()]*D)
    s = jnp.array([sx.flatten() for sx in ss]).T

    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    #print(knot_y.shape)
    
    prod = (design_space @ s.T).T
    #print(prod.shape)
    
    # compute the conjugate
    lft1 = jnp.max(prod[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    #print(lft1.shape)
    # compute the biconjugate
    lft2 = jnp.max(prod[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal
    
    return lft2


def generate_true_function(design_space, knot_N): # todo: pass kernel as argument
    # TODO: make it such that we can pass in a dimension too? or just stick to 1d vs 2d. 
    #knot_x = jnp.linspace(0, 1, knot_N)
    knot_K = K(kernel_rbf, design_space, design_space) + 1e-8 * jnp.eye(knot_N)
    # Cholesky decomposition of the kernel matrix
    knot_cK = spla.cholesky(knot_K)
    # Form the true function of interest at knot_N number of points
    true_y = knot_cK.T @ npr.randn(knot_N)
    # compute envelope based on true function
    true_envelope = convelope(design_space, true_y)
    
    return true_y, true_envelope


def compute_IG_putative_x(putative_x, design_space, data, pred_cK, pred_Y, tights, rng_key, T=100, J=200):
    """
    Compute a Monte Carlo approximation of the IG w.r.t. T samples of s_t ~ p(s | data).
    
    The inner entropy is approximated via Monte Carlo + a KDE estimator constructed from the samples. 
    (TODO: don't use the same data twice)
    
    T: number of samples for the outer expectation
    J: number of ESS samples (controls the # of samples for the inner MC too?)
    """

    def entropy_est_wrap(args):
        tights_i, pred_Y_i = args
        return ess_and_estimate_entropy(putative_x, design_space, data, tights_i, pred_Y_i, pred_cK, rng_key, J=J)
    ventropy_est = jax.jit(jax.vmap(entropy_est_wrap, in_axes=((1,1),)))
    
    entropies = ventropy_est((tights, pred_Y))  
    #entropies = jnp.array([ess_and_estimate_entropy(putative_x, design_space, data, tights[:,i], pred_Y[:,i], pred_cK, rng_key, J=J) for i in range(len(tights))])
    
    # estimate of the second term in the EIG
    return entropies.mean()
    

def get_next_candidate(data, design_inds, design_space, rng_key, T=30, J=40, tol=1e-3):
    """
    Given current data and a list of designs, computes an IG score for each design. 
    
    T: number of outer MC samples
    J: number of inner MC samples
    tol: tolerance for considering what is tight w.r.t. the convex hull
    
    Returns the best design and the list of scores. 
    """
    
    #print(design_inds)

    # updates the model and samples T functions and computes their envelopes. here we evaluate functions only at points in the design space
    pred_mean, pred_cov, pred_cK, pred_Y, envelopes = update_model(data, T, design_space) 
    # compute the vector of indicators
    tights = jnp.abs(envelopes.T - pred_Y) < tol ## NOTE: we transposed the shape from what it was before
    
    # TODO: move the lambda function into the vmap to make it cleaner
    compute_IG_putative_wrap = lambda x: compute_IG_putative_x(x, design_space, data, pred_cK, pred_Y, tights, rng_key, T = T, J = J) 
    compute_IG_vmap = jax.jit(jax.vmap(compute_IG_putative_wrap, in_axes=0))
    
    #curr_entropy = jnp.log(jnp.sqrt(2*jnp.pi*jnp.e*jnp.diag(pred_cov[jnp.ix_(design_inds, design_inds)]))) # TOD0: need to index the relevant designs
    curr_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.diag(pred_cov)) 
    mean_entropy = compute_IG_vmap(design_space[design_inds,:])
    
    entropy_change = curr_entropy - mean_entropy
    return entropy_change.argmax(), entropy_change




def plot_candidate(knot_x, true_y, true_envelope, pred_mean, pred_cov, envelopes, data, next_x, scores, plot_eig=True, legend=True, plot_hulls=True):
    #train_x, train_y = data
    
    plt.figure(figsize=(10,4))

    # Plot true function, convex hull, and GP model mean + uncertainty
    plt.plot(knot_x, true_y, "k", lw=2, label="True function")
    plt.plot(knot_x, true_envelope.T, ls="dashed", label="True envelope", lw=2, c="k")
    plt.plot(knot_x, pred_mean, lw=2, c="tab:blue", label="Model prediction")
    y_err = 2 * jnp.sqrt(jnp.diag(pred_cov))
    plt.fill_between(knot_x, pred_mean - y_err, pred_mean + y_err, alpha=0.4, color="tab:blue")
    
    if plot_hulls:
        # Plot convex hulls of posterior samples
        plt.plot(knot_x, envelopes[:15,:].T, lw=0.5, c="gray")

    # Plot data / next evaluation
    plt.scatter(next_x, get_next_y(true_y, knot_x[:,jnp.newaxis], next_x), marker="*", color="tab:red", zorder=5, sizes=[150], label="Next evaluation")
    plt.scatter(dataset.X, dataset.y, label="Observed data", c="k", zorder=5)

    # Plot the entropy estimates (i.e., -second term of EIG)
    if plot_eig:
        plt.scatter(knot_x, scores, c="purple", marker="|", label="EIG")
        
    if legend:
        plt.legend(ncol=2)

    plt.ylim(-3,3); plt.xlabel("Composition space"); plt.ylabel("Energy")
    


def compute_distances(data, true_envelopes):
    distances = []
    for (i,x) in enumerate(data[0]):
        x_idx = (knot_x == x).argmax()
        distances.append(jnp.abs(data[1][i] - true_envelope[1][x_idx]))
    return jnp.array(distances)

def get_next_y(true_y, design_space, next_x):
    return true_y[:,jnp.newaxis][design_space == next_x]


def plot_candidate_old(knot_x, true_y, true_envelope, pred_mean, pred_cov, envelopes, data, next_x, scores, plot_eig=True, legend=True, plot_hulls=True):
    train_x, train_y = data
    
    plt.figure(figsize=(10,4))

    # Plot true function, convex hull, and GP model mean + uncertainty
    plt.plot(knot_x, true_y, "k", lw=2, label="True function")
    plt.plot(knot_x, true_envelope.T, ls="dashed", label="True envelope", lw=2, c="k")
    plt.plot(knot_x, pred_mean, lw=2, c="tab:blue", label="Model")
    y_err = jnp.sqrt(jnp.diag(pred_cov))
    plt.fill_between(knot_x, pred_mean - y_err, pred_mean + y_err, alpha=0.4, color="tab:blue")
    
    if plot_hulls:
        # Plot convex hulls of posterior samples
        plt.plot(knot_x, envelopes[:15,:].T, lw=0.5, c="gray")

    # Plot data / next evaluation
    plt.scatter(knot_x[next_x], true_y[next_x], marker="*", color="tab:red", zorder=5, sizes=[150], label="Next evaluation")
    plt.scatter(train_x, train_y, label="Observed data", c="k", zorder=5)

    # Plot the entropy estimates (i.e., -second term of EIG)
    if plot_eig:
        plt.scatter(knot_x, scores, c="purple", marker="|", label="EIG")
        
    if legend:
        plt.legend(ncol=2)

    plt.ylim(-3,3); plt.xlabel("Composition space"); plt.ylabel("Energy")
    


def plot_triangle(pts, tight_pts, data, next_x = None, plot_design = True):
    """
    Plot for the 2-simplex examples on a triangle.
    """
    import ternary
    fontsize=20
    ### Scatter Plot
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    #tax.set_title("", fontsize=20)
    tax.boundary(linewidth=2.0)
    #tax.gridlines(multiple=1, color="blue")
    
    x_train_pts = []
    for x in data[0]:
        x_train_pts.append((x[0], x[1], 1-x.sum()))

    if plot_design:
        tax.scatter(pts, marker=".", label="Design space",  sizes=jnp.ones(len(pts))*150, c="gray", zorder=3)
    tax.scatter(x_train_pts, marker=".", label="Observation", sizes=jnp.ones(len(pts))*250, zorder=3, c="k")
    tax.scatter(tight_pts, marker="*", label="Tight points", sizes=jnp.ones(len(pts))*250, zorder=5, c="tab:blue", alpha=0.5)
    
    if next_x != None:
        tax.scatter([pts[next_x]], marker="*", label="Tight points", sizes=jnp.ones(len(pts))*250, zorder=6, c="tab:red")

    tax.legend(loc="upper right")
    tax.right_corner_label("B", fontsize=fontsize)
    tax.top_corner_label("A", fontsize=fontsize)
    tax.left_corner_label("C", fontsize=fontsize)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=1, multiple=1, offset=0.02, fontsize=12)
    tax.show()