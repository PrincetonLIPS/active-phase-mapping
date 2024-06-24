import jax
import jax.numpy as jnp
import jax.random as jrnd

from functools import partial

from .kernels import Matern52


@partial(jax.jit, static_argnums=(4,5,6))
def log_marginal_likelihood(
    X, y,
    lengthscales,
    amplitude,
    kernel_fn,
    jitter,
    noise_missing,      
  ):
  ''' Compute the log marginal likelihood for the GP for a single phase.

  Missing observations, which might arise due to only evaluating a subset of
  the phases, are represented by nan values in Y. The noise for these missing
  values is set to noise_missing.  When this is large, the log marginal
  likelihood will be essentially the same as what we would get without those
  observations. We do it this way to be friendly to vmap; it's essentially
  padding.

  Args:
    X: The input data. [num_data x num_species]
    Y: The output data. [num_data] nan for missing values.
    lengthscales: The length scales. [num_species]
    amplitudes: The amplitude.
    kernel_fn: The kernel function.
    jitter: The jitter.
    noise_missing: The noise for missing values.

  Returns:
    log_marginal_likelihood: The log marginal likelihood.
  '''
  def apply_kernel(X1, X2, ls, amp):
    return amp**2 * kernel_fn(X1/ls, X2/ls)

  # Zero out the missing values and make the noise for them big.
  missing = jnp.isnan(y)
  y = jnp.where(missing, jnp.zeros_like(y), y)
  noise = jnp.where(missing, noise_missing, jitter)

  # TODO: subtract off the mean, accounting for missing data.

  # Compute the kernel matrix for the observed values.
  K_X_X = apply_kernel(X, X, lengthscales, amplitude) + jnp.diag(noise)

  # Compute the Cholesky factor of the kernel matrix.
  # This is upper triangular.
  chol_K_X_X = jax.scipy.linalg.cholesky(K_X_X)

  # Solve the linear system.
  solved = jax.scipy.linalg.cho_solve((chol_K_X_X, False), y)

  # Compute the log marginal likelihood.
  lml = -0.5 * jnp.dot(y, solved) \
    - jnp.sum(jnp.log(jnp.diag(chol_K_X_X))) \
    - 0.5 * X.shape[0] * jnp.log(2*jnp.pi)
  
  return lml