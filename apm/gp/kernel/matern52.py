from typing import Optional

import jax
import jax.numpy as jnp

def Matern52(X: jax.Array, Y: Optional[jax.Array]=None) -> jax.Array:
  ''' Compute the Matern 5/2 kernel function between two sets of points, or
      within a single set of points.

      Args:
        X: The first set of points.
        Y: The second set of points. If None, computes the kernel function
           within X.

      Returns:
        The kernel function between the points in X and Y.
  '''
  r2 = sqdist(X, Y)
  r = jnp.sqrt(r2)
  return (1.0 + jnp.sqrt(5.0) * r + 5.0 / 3.0 * r2) \
    * jnp.exp(-jnp.sqrt(5.0) * r)

