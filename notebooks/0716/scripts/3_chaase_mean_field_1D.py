import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from copy import deepcopy
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import imageio

#JAX
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jaxutils import Dataset

# Imports from our code base
from active_search import get_next_y, get_next_candidate_baseline
from drew_nd import *
from drew_plot import *
from gp_model import update_model, make_preds
from search_no_gpjax import generate_true_function

#set seed
seed = 5
npr.seed(seed); rng_key = jrnd.PRNGKey(seed)

###Design parameters
n_grid=40
dimensions=2
iterations=10
num_polymorphs=2
###Sampling parameters
num_y, num_samples, num_curves = (10, 100, 30)
#Master dictionary
poly_dict={}
for i in range(num_polymorphs):
    poly_dict[i]=deepcopy({'true_y':[],'design_space':[],'pred_mean':[],
    'pred_cov':[],'posterior':[],'params':[],'train_x':[],'train_y':[],
    'dataset':[]})

#Producing grid
pts=nD_coordinates(dimensions,0,1,n_grid)
knot_x = jnp.linspace(0, 1, n_grid)
design_space = knot_x[:, jnp.newaxis]
knot_N = len(design_space)
print(knot_N)
endpoint_indices=[-1,0]

#Saving energies and GP models
for i in range(num_polymorphs):
    poly_dict[i]['true_y'] = generate_true_function(design_space, knot_N)
    poly_dict[i]['train_x'] = jnp.array([design_space[0], design_space[-1]])
    poly_dict[i]['train_y'] = jnp.array([poly_dict[i]['true_y'][0], poly_dict[i]['true_y'][-1]])
    poly_dict[i]['dataset'] = Dataset(X=poly_dict[i]['train_x'], y=poly_dict[i]['train_y'][:,jnp.newaxis])
    poly_dict[i]['designs'] = deepcopy(design_space[1:-1])
    poly_dict[i]['pred_mean'], poly_dict[i]['pred_cov'], poly_dict[i]['posterior'], poly_dict[i]['params'] = update_model(
    poly_dict[i]['dataset'], design_space, rng_key, update_params=False) # Update the model given the data above

#Determining true_hull
all_true_curves=[]
for i in range(num_polymorphs):
    all_true_curves.append(poly_dict[i]['true_y'])
min_curve=np.min(np.vstack(all_true_curves),axis=0)
#Set endpoints to zero.
lin_comb=get_lin_comb(pts,endpoint_indices,min_curve)
Y_zeroed=min_curve-lin_comb
#Generating hull
tmp=np.column_stack([np.array(design_space.ravel()),Y_zeroed])
points=tmp[tmp[:,1]<=0]
true_e_hull=get_hull_energies_oneD(design_space,Y_zeroed,endpoint_indices=endpoint_indices) + lin_comb

#Generating dictionaries for indexing. Dicts are necessary since indices will change as we delete designs.
for i in range(num_polymorphs):
    poly_dict[i]['true_y_dict']={} #energy_dict to query from.
    for comp, y in zip(design_space,poly_dict[i]['true_y']):
        poly_dict[i]['true_y_dict'][float(comp)]=y
index_dict={}
for index,x in enumerate(design_space):
    #comp_tuple=tuple([float(i) for i in x])
    index_dict[float(x)]=index

###Active search
means=[]
stds=[]
rmses=[]
for iteration in tqdm(range(iterations)):
    print("Iteration: ", iteration)
    plot_search(knot_x, poly_dict, design_space, 100, num_polymorphs, 100, pts, endpoint_indices, iteration, true_e_hull)
    #def plot_search(knot_x, poly_dict, design_space, num_samples, num_polymorphs, num_curves, pts, endpoint_indices, iteration, true_e_hull):

    #Quantifying error
    hull_expected_energy, initial_entropy=calc_expected_energy_or_entropy(poly_dict=poly_dict, design_space=design_space, num_curves=num_curves, num_samples=num_samples, knot_N=knot_N, pts=pts, endpoint_indices=endpoint_indices)
    errors=jnp.abs(true_e_hull-hull_expected_energy)
    means.append(jnp.mean(errors))
    stds.append(jnp.std(errors))
    print(means)

    #Calculate EIG for each polymorph and over all compositions.
    data_dict = {} #dictionary mapping EIG to composition
    for i in range(num_polymorphs):
        data_dict[i]=deepcopy({})
        for composition in tqdm(poly_dict[i]['designs']):
            EIG=EIG_chaase_meanfield_multi(composition, poly_dict[i]['dataset'], seed, index_dict,  poly_dict[i]['pred_mean'], poly_dict[i]['pred_cov'],
            design_space, num_y, initial_entropy, pts, endpoint_indices, knot_N, num_curves, num_samples, poly_dict, i, num_polymorphs)
            data_dict[i][float(EIG)] = composition

    #choosing next_x
    maxs={}
    for i in range(num_polymorphs):
        maxs[np.max(list(data_dict[i].keys()))]=i
    print(maxs)
    max_EIG=np.max(list(maxs.keys()))
    max_polymorph=maxs[max_EIG]
    next_x=data_dict[max_polymorph][max_EIG]

    #obtain y-value corresponding to next_x
    #next_x_tuple=tuple([float(i) for i in next_x]) #converting composition into tuple.
    next_y=poly_dict[max_polymorph]['true_y_dict'][float(next_x)]
    #updating model with composition and y-value
    poly_dict[max_polymorph]['dataset'] = poly_dict[max_polymorph]['dataset'] + Dataset(X=jnp.atleast_2d(next_x), y=jnp.atleast_2d(next_y))
    poly_dict[max_polymorph]['designs'] = jnp.delete(poly_dict[max_polymorph]['designs'], (poly_dict[max_polymorph]['designs'] == next_x).argmax())[:, jnp.newaxis]
    poly_dict[max_polymorph]['pred_mean'], poly_dict[max_polymorph]['pred_cov'], poly_dict[max_polymorph]['posterior'], poly_dict[max_polymorph]['params'] = update_model(poly_dict[max_polymorph]['dataset'], design_space, rng_key, update_params=False)

df=pd.DataFrame()
df['means']=means
df['stds']=stds
df['iteration']=[i for i in range(iterations)]
df.to_csv(f'../data/chaase_meanfield_{num_y}_{num_samples}.csv')

#Producing GIF of search.
frames = []
for i in range(iterations):
    image = imageio.v2.imread(f'../figs/{i}_{num_polymorphs}.png')
    frames.append(image)

imageio.mimsave(f'../figs/polymorph_search_{num_polymorphs}.gif', # output gif
                frames,          # array of input frames
                duration=2000,
                loop=5)
