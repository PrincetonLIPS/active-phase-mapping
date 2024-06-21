import jax
import jax.numpy as jnp
import numpy as np

from scipy.spatial import ConvexHull


def numpy_lower_hull_points(X, y):
  ''' Return boolean values about which points are tight. '''

  # Concatenate the X and y values.
  X = jax.numpy.concatenate((X, y[:,None]), axis=1)

  # Get the convex hull.
  hull = ConvexHull(X)

  # Identify downward-facing facets based on the last coordinate of the normal.
  downward_facets = hull.equations[:,-2] < 0

  # Get the unique set of vertices associated with the downward-facing facets.
  tight = np.isin(np.arange(X.shape[0]), hull.simplices[downward_facets])

  return tight

def lower_hull_points(X, y):
  result_shape = jax.ShapeDtypeStruct((X.shape[0],), jnp.bool_)
  return jax.pure_callback(numpy_lower_hull_points, result_shape, X, y)

if __name__ == "__main__":

  rng = jax.random.PRNGKey(4)
  x_rng, y_rng = jax.random.split(rng, 2)
  X = jax.random.uniform(x_rng, (50,1))
  y = jax.random.normal(y_rng, (50,))

  tight = lower_hull_points(X, y)
  print(tight)

  import matplotlib.pyplot as plt
  plt.plot(X, y, 'o')
  plt.plot(X[tight], y[tight], 'x')
  plt.savefig("hull-test-1d.pdf")