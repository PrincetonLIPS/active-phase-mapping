import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla
from jax.config import config
config.update("jax_enable_x64", True)
import numpy.random as npr

def kernel_rbf(x1, x2):
    """
    Squared expoential kernel with lengthscale ls.
    """
    ls=0.1; v=1
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

def generate_true_function(design_space, knot_N): # todo: pass kernel as argument

    # TODO: make it such that we can pass in a dimension too? or just stick to 1d vs 2d.
    knot_x = jnp.linspace(0, 1, knot_N)

    knot_K = K(kernel_rbf, design_space, design_space) + 1e-8 * jnp.eye(knot_N)
    # Cholesky decomposition of the kernel matrix
    knot_cK = spla.cholesky(knot_K)
    # Form the true function of interest at knot_N number of points
    true_y = knot_cK.T @ npr.randn(knot_N)

    return true_y
