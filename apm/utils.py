import jax
import jax.numpy as jnp

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

