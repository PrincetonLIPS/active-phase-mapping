import jax
import jax.numpy as jnp
import jax.random as jrnd

from functools import partial

from .kernels import Matern52

@partial(jax.jit, static_argnums=(5,6,7))
def _posterior_params_phase(
    X, y,
    qX,
    lengthscales,
    amplitude,
    kernel_fn,
    jitter,
    noise_missing,
  ):
  def apply_kernel(X1, X2, ls, amp):
    return amp * kernel_fn(X1/ls, X2/ls)

  # Zero out the missing values and make the noise for them big.
  missing = jnp.isnan(y)
  y = jnp.where(missing, jnp.zeros_like(y), y)
  noise = jnp.where(missing, noise_missing, jitter)

  # Compute the kernel matrix for the observed values.
  K_X_X = apply_kernel(X, X, lengthscales, amplitude) + jnp.diag(noise)

  # Compute the kernel matrix for the query points.
  K_qX_qX = apply_kernel(qX, qX, lengthscales, amplitude)

  # Compute the kernel matrix between the query and observed points.
  K_qX_X = apply_kernel(qX, X, lengthscales, amplitude)

  # Compute the Cholesky factor of the kernel matrix.
  chol_K_X_X = jax.scipy.linalg.cho_factor(K_X_X)

  # Solve the linear system with the cross-covariance.
  solved = jax.scipy.linalg.cho_solve(chol_K_X_X, K_qX_X.T)

  # Compute the posterior mean.
  mean = jnp.dot(solved.T, y)

  # Compute the posterior covariance.
  cov = K_qX_qX - jnp.dot(solved.T, K_qX_X) + jitter*jnp.eye(qX.shape[0])

  return mean, cov
  
@partial(jax.jit, static_argnums=(5,6,7))
def posterior_params(
    X, Y,
    qX,
    lengthscales,
    amplitudes,
    kernel_fn,
    jitter,
    noise_missing,
  ):
  ''' Compute the posterior mean and covariance for each phase.

  Missing observations, which might arise due to only evaluating a subset of
  the phases, are represented by nan values in Y. The noise for these missing
  values is set to noise_missing.  When this is large, the posterior mean will
  be essentially the same as what would be predicted without that observation.
  We do it this way to be friendly to vmap; it's essentially padding.

  Args:
    X: The input data. [num_data x num_species]
    Y: The output data. [num_data x num_phases] nan for missing values.
    qX: The query data. [num_query x num_species]
    lengthscales: The length scales. [num_phases x num_species]
    amplitudes: The amplitudes. [num_phases]
    kernel_fn: The kernel function.
    jitter: The jitter.
    noise_missing: The noise for missing values.

  Returns:
    means: The posterior means. [num_query x num_phases]
    covs: The posterior covariances. [num_query x num_query x num_phases]
  '''
  return jax.vmap(
      _posterior_params_phase,
      in_axes=(0, 0, 0, 0, None, None, None),
    )(X, Y, qX, lengthscales, amplitudes, kernel_fn, jitter, noise_missing)

