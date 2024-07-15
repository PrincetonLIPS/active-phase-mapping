from typing import Tuple
from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp

def rff_basis_function(
    x:               ArrayLike, # num_species [float32]
    rff_projections: ArrayLike, # num_rff x num_species [float32]
    rff_phases:      ArrayLike, # num_rff [float32]
    lengthscales:    ArrayLike, # num_species [float32]
    amplitude:       ArrayLike, # [float32]
  ) -> Array: # num_rff [float32]

  num_rff = rff_projections.shape[0]
  rff = jnp.sqrt(2.0 / num_rff) * jnp.cos(
      jnp.dot(rff_projections, x/lengthscales) + rff_phases
    )
  assert rff.shape == (num_rff,)
  return amplitude * rff.squeeze()

@jax.jit
def weight_posterior_cholesky(
    basis_X:     ArrayLike, # num_data x num_rff [float32]
    y:           ArrayLike, # num_data [float32]
    noise:       ArrayLike, # num_data [boolean]
    prior_mean:  ArrayLike, # num_rff [float32]
    prior_chol:  ArrayLike, # num_rff x num_rff [float32]
  ) -> Tuple[Array, Array]: # num_rff [float32], num_rff x num_rff [float32]
  # TODO: explore an SVD alternative

  assert basis_X.shape[1] == prior_mean.shape[0]
  assert basis_X.shape[1] == prior_chol.shape[0]
  assert basis_X.shape[1] == prior_chol.shape[1]
  assert basis_X.shape[0] == y.shape[0]
  assert y.shape == noise.shape

  num_data = basis_X.shape[0]
  num_rff  = basis_X.shape[1]

  # Find the inverse of the prior using its Cholesky decomposition.
  prior_iSigma = jax.scipy.linalg.cho_solve((prior_chol, False), jnp.eye(num_rff))
  assert prior_iSigma.shape == (num_rff, num_rff)

  # Compute the inverse of the posterior covariance.
  iSigma = prior_iSigma + jnp.dot(basis_X.T, (1/noise[:,jnp.newaxis]) * basis_X)
  assert iSigma.shape == (num_rff, num_rff)

  # Take the Cholesky of the inverse to get the posterior covariance.
  chol_iSigma = jax.scipy.linalg.cholesky(iSigma)

  # FIXME there must be a better way to do this.
  # For some reason solve_triangular with identity on chol_iSigma doesn't
  # do the right thing.
  #chol_Sigma = jax.scipy.linalg.solve_triangular(chol_iSigma, jnp.eye(cfg.gp.num_rff))
  Sigma = jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(num_rff))
  chol_Sigma = jax.scipy.linalg.cholesky(Sigma)

  to_solve = jnp.dot(basis_X.T, y/noise) + jnp.dot(prior_iSigma, prior_mean)
  assert to_solve.shape == (num_rff,)

  #mu = Sigma @ to_solve
  mu = jax.scipy.linalg.cho_solve((chol_iSigma, False), to_solve)
  assert mu.shape == (num_rff,)

  return mu, chol_Sigma