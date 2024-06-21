import jax
import jax.numpy as jnp

from jaxhull.hull import convex_hull as jh_convex_hull

@jax.jit
def lower_hull_points(X, y):
  ''' Return boolean values about which points are tight. '''

  # Concatenate the X and y values.
  X = jax.numpy.concatenate((X, y[:,None]), axis=1)

  # Get the convex hull.
  hull = jh_convex_hull(X, jnp.ones(X.shape[0], dtype=bool))

  # Identify downward-facing facets.
  # Ignore unused facets.
  downward = jnp.logical_and(hull.normals[:,-1] < 0, hull.facets[:,0] >= 0)

  # Make all the facet vertices -1 for non-downward-facing facets.
  downward_facets = jnp.where(downward[:,jnp.newaxis], hull.facets, -1)

  # Boolean mask on whether the vertices are in a downward-facing facet.
  tight = jnp.any(downward_facets[jnp.newaxis,:,:] == jnp.arange(X.shape[0])[:,jnp.newaxis,jnp.newaxis], axis=(1,2))

  return tight

if __name__ == "__main__":

  rng = jax.random.PRNGKey(2)
  x_rng, y_rng = jax.random.split(rng, 2)
  X = jax.random.uniform(x_rng, (50,1))
  y = jax.random.normal(y_rng, (50,))

  tight = lower_hull_points(X, y)
  print(tight)

  import matplotlib.pyplot as plt
  plt.plot(X, y, 'o')
  plt.plot(X[tight], y[tight], 'x')
  plt.savefig("hull-test-1d.pdf")