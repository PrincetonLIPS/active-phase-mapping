import numpy.random as npr
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
from scipy.spatial import ConvexHull

from active_search import get_next_y, get_next_candidate_baseline, get_next_candidate, compute_distances
from gp_model import update_model
from search_no_gpjax import generate_true_function
from drew_nd import *

#set seed
seed = 3
npr.seed(seed); rng_key = jrnd.PRNGKey(seed)

n_grid, dimensions=(11,4)

#Producing grid
pts=nD_coordinates(dimensions,0,1,n_grid)
design_space=np.array(pts)[:,:dimensions-1]
knot_N = len(design_space)
print(knot_N)

# generate energies
true_y=generate_true_function(design_space, knot_N)
endpoint_indices=get_endpoint_indices(dimensions,pts)

#Set endpoints to zero.
lin_comb=get_lin_comb(pts,endpoint_indices,true_y)
Y_zeroed=true_y-lin_comb

#Generating hull
points=np.column_stack((design_space,Y_zeroed))[Y_zeroed<=0]

#Determining vertices
E_above_hull=Y_zeroed-get_hull_energies(design_space,Y_zeroed,endpoint_indices=endpoint_indices)
tol=1e-3
vertices=E_above_hull<tol
true_hull=jnp.zeros(knot_N).at[vertices].set(1)
