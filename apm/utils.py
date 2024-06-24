from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrnd

from itertools import combinations

def stars_and_bars(n, k):
  ''' Generate all the possible ways that n items can be distributed into k bins. 
  
  Args:
    n: int, number of items.
    k: int, number of bins.

  Returns:
    jnp.ndarray, shape (C(n+k-1, k-1), k), where C(n, k) is the binomial coefficient.
      Each column represents the number of items in each bin.
  '''
  bar_indices = jnp.sort(jnp.array(list(combinations(range(n+k-1), k-1))), axis=1)
  buckets = jnp.diff(bar_indices - jnp.arange(k-1), axis=1, prepend=0)
  return jnp.hstack([
    buckets,
    n + k - 2 - bar_indices[:, -1][:, jnp.newaxis],
  ])

def sqdist(X: jax.Array, Y: Optional[jax.Array]=None) -> jax.Array:
  ''' Compute the squared distance between two sets of points, or within a
      single set of points.

      Args:
        X: The first set of points.
        Y: The second set of points. If None, computes the squared distance
           within X.

      Returns:
        The squared distance between the points in X and Y.
  '''
  if Y is None:
    sumX: jax.Array = jnp.sum(X**2, axis=-1)
    return jnp.abs(sumX[:, jnp.newaxis] + sumX[jnp.newaxis, :] - 2 * jnp.dot(X, X.T))
  else:
    return jnp.abs(jnp.sum(X**2, axis=-1)[:, jnp.newaxis] \
      + jnp.sum(Y**2, axis=-1)[jnp.newaxis, :] \
      - 2 * jnp.dot(X, Y.T))
  
def multivariate_t_rvs(rng, mu, sigma, df, shape):
  ''' Generate random samples from a multivariate t-distribution.

  Args:
    rng: PRNGKey, the random number generator key.
    mu: jnp.ndarray, shape (d,), the mean of the distribution.
    sigma: jnp.ndarray, shape (d, d), the scale matrix of the distribution.
    df: float, the degrees of freedom of the distribution.
    shape: tuple, the shape of the output.

  Returns:
    jnp.ndarray, shape (shape + (d,)), the random samples.
  '''
  chi_rng, norm_rng = jrnd.split(rng)

  # Generate zero-mean multivariate normal samples with covariance Sigma.
  Z = jrnd.multivariate_normal(
    norm_rng,
    mean=jnp.zeros_like(mu),
    cov=sigma,
    shape=shape,
  )

  # Generate chi-squared random variables.
  chi2 = jrnd.chisquare(
    chi_rng,
    df,
    shape=shape,
  )

  return mu + Z * jnp.sqrt(df / chi2)[..., jnp.newaxis]
