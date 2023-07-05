import numpy.random as npr
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jaxutils import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_search import get_next_y, get_next_candidate_baseline, get_next_candidate, compute_distances
from gp_model import update_model
from search_no_gpjax import generate_true_function, sample_from_posterior
import matplotlib.cm as cm
from matplotlib import rcParams
from drew_nd import get_lin_comb, elaborate_point, get_hull_energies
import numpy as np
from scipy.spatial import ConvexHull

##Setting up the problem

seed = 2 # seed to use for all expts
num_iters = 14 # number of iterations to run
knot_N = 40 # number of points to discretize space into

# grid to discretize design space into
knot_x = jnp.linspace(0, 1, knot_N) #composition axis
design_space = knot_x[:, jnp.newaxis] #Transposes? Converts from one long 1D array to array of arrays.

npr.seed(seed); rng_key = jrnd.PRNGKey(seed) #Random seed.

# generate true data and envelope
true_y, true_envelope = generate_true_function(design_space, knot_N) #Using some function with a kernel. Also, generating convex hull using convelope function.

endpoint_indices=[-1,0]
#Adding implied coordinate. -- Going from 0.3 to [0.3,0.7]
pts=[]
for x in design_space:
    pts.append(elaborate_point(x))

#Zeroing endpoints
lin_comb=get_lin_comb(pts,endpoint_indices,true_y)
Y_zeroed=true_y-lin_comb

#Building hull
points=np.column_stack((design_space,Y_zeroed))[Y_zeroed<=0]#Filtering out all positive values. May want to think about 0-values in future.
hull=ConvexHull(points)

#Determining vertices
hull_energies=get_hull_energies(design_space,Y_zeroed,endpoint_indices=endpoint_indices)
E_above_hull=Y_zeroed-hull_energies
tol=1e-3
vertices=E_above_hull<tol
true_hull=jnp.zeros(knot_N).at[vertices].set(1)


#Plotting
#Convelope involves that weird conjugate biconjugate business.
size=14
rcParams['font.family'] = 'Arial'

#build Y_non-positive (for plotting purposes)
y_nonpos=[]
for x in Y_zeroed:
    if x<=0:
        y_nonpos.append(x)
    else:
        y_nonpos.append(0)

#First fig
fig = plt.figure()
ax = fig.add_subplot(111)

#plt.plot(knot_x,true_envelope[0],ls='-.',linewidth=3,color='#5d5d5d')
#plt.plot(knot_x,true_y,ls='-',linewidth=3,color='black')
plt.plot(knot_x,hull_energies,ls='-',linewidth=3,color='#5d5d5d',zorder=2)
plt.plot(knot_x,np.array(y_nonpos),ls='-',linewidth=3,zorder=1)

plt.xlabel('Composition',size=size)
plt.ylabel('Energy',size=size)
#plt.ylim(-1.5,-0.2)

plt.tick_params(direction='in',which='both',labelsize = size)
plt.tight_layout()
ax.set_aspect(1.0/ax.get_data_ratio())
#plt.show()
plt.savefig('../figs/hull_energy.png',dpi=600)
