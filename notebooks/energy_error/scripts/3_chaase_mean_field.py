import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from copy import deepcopy

#JAX
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jaxutils import Dataset

# Imports from our code base
from active_search import get_next_y, get_next_candidate_baseline
from drew_nd import *
from gp_model import update_model, make_preds
from search_no_gpjax import generate_true_function

#set seed
seed = 3
npr.seed(seed); rng_key = jrnd.PRNGKey(seed)

n_grid, dimensions, iterations=(11, 3, 20)

#Producing grid
pts=nD_coordinates(dimensions,0,1,n_grid)
design_space=np.array(pts)[:,:dimensions-1]
knot_N = len(design_space)
print(knot_N)

# generate energies
true_y=generate_true_function(design_space, knot_N)
endpoint_indices=get_endpoint_indices(dimensions,pts)

#energy_dict to query from.
true_y_dict={}
for comp,y in zip(design_space,true_y):
    comp=tuple([float(i) for i in comp])
    true_y_dict[comp]=y

#Training on endpoint indices.
x_lst,y_lst=[],[]
for endpoint_index in endpoint_indices:
    x_lst.append(design_space[endpoint_index])
    y_lst.append(true_y[endpoint_index])
train_x,train_y=jnp.array(x_lst),jnp.array(y_lst)
data = (train_x, train_y); dataset = Dataset(X=train_x, y=train_y[:,jnp.newaxis])
#removing endpoint indices from list of candidates.
designs = [x for index,x in enumerate(design_space) if index not in endpoint_indices]

# Update the model given the data above
pred_mean, pred_cov, posterior, params = update_model(dataset, design_space, rng_key, update_params=False)

###Determining true_hull
#Set endpoints to zero.
lin_comb=get_lin_comb(pts,endpoint_indices,true_y)
Y_zeroed=true_y-lin_comb
#Generating hull
points=np.column_stack((design_space,Y_zeroed))[Y_zeroed<=0]
#Determining vertices
true_e_hull=get_hull_energies(design_space,Y_zeroed,endpoint_indices=endpoint_indices)


index_dict={}
for index,x in enumerate(design_space):
    comp_tuple=tuple([float(i) for i in x])
    index_dict[comp_tuple]=index

###Active search
means=[]
stds=[]
initial_entropies=[]

num_y, num_samples = (5, 5)

for i in tqdm(range(iterations)):
    print("Iteration: ", i)

    #Quantifying error
    hull_expected_energy, initial_entropy=calc_expected_energy_or_entropy(
    design_space=design_space, num_samples=num_samples, knot_N=knot_N, pts=pts, endpoint_indices=endpoint_indices,
    pred_mean=pred_mean, pred_cov=pred_cov)
    errors=jnp.abs(true_e_hull-hull_expected_energy)
    means.append(jnp.mean(errors))
    stds.append(jnp.std(errors))
    print(means)

    #Calculate EIG for each composition.
    data_dict = {} #dictionary mapping EIG to composition
    for composition in designs:
        EIG=EIG_chaase_meanfield(composition, dataset, seed, index_dict,  pred_mean, pred_cov,
        design_space, num_y, initial_entropy, pts, endpoint_indices, knot_N, num_samples)
        data_dict[float(EIG)] = composition


    #choosing next_x
    max_EIG=np.max(list(data_dict.keys())) #find max EIG
    next_x=data_dict[max_EIG] #find corresponding composition
    #obtain y-value corresponding to next_x
    next_x_tuple=tuple([float(i) for i in next_x]) #converting composition into tuple.
    next_y=true_y_dict[next_x_tuple]
    #updating model with composition and y-value
    dataset = dataset + Dataset(X=jnp.atleast_2d(next_x), y=jnp.atleast_2d(next_y)) #updating dataset with new value
    pred_mean, pred_cov, posterior, params = update_model(dataset, design_space, rng_key, update_params=False) #update the model.
    designs = jnp.delete(jnp.array(designs), (jnp.array(designs) == next_x).sum(1).argmax(), axis=0)

    print(len(designs))

df=pd.DataFrame()
df['means']=means
df['stds']=stds
df['iteration']=[i for i in range(iterations)]
df.to_csv(f'../data/chaase_meanfield_{num_y}_{num_samples}.csv')
