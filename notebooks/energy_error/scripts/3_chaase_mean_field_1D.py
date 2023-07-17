import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from copy import deepcopy
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.pyplot as plt

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

n_grid, dimensions, iterations=(40, 2, 3)
pts=nD_coordinates(dimensions,0,1,n_grid)

#Producing grid
knot_x = jnp.linspace(0, 1, n_grid)
design_space = knot_x[:, jnp.newaxis]
knot_N = len(design_space)
print(knot_N)

# generate energies
true_y=generate_true_function(design_space, knot_N)
#endpoint_indices=get_endpoint_indices(dimensions,pts)
endpoint_indices=[-1,0]


train_x = jnp.array([design_space[0], design_space[-1]]); train_y = jnp.array([true_y[0], true_y[-1]])
dataset = Dataset(X=train_x, y=train_y[:,jnp.newaxis])
designs = design_space[1:-1]


# Update the model given the data above
pred_mean, pred_cov, posterior, params = update_model(dataset, design_space, rng_key, update_params=False)

###Determining true_hull
#Set endpoints to zero.
lin_comb=get_lin_comb(pts,endpoint_indices,true_y)
Y_zeroed=true_y-lin_comb
#Generating hull

tmp=np.column_stack([np.array(design_space.ravel()),Y_zeroed])
points=tmp[tmp[:,1]<=0]

#Determining vertices
true_e_hull=get_hull_energies_oneD(design_space,Y_zeroed,endpoint_indices=endpoint_indices)


index_dict={}
for index,x in enumerate(design_space):
    #comp_tuple=tuple([float(i) for i in x])
    index_dict[float(x)]=index

#energy_dict to query from.
true_y_dict={}
for comp,y in zip(design_space,true_y):
    true_y_dict[float(comp)]=y


###Active search
means=[]
stds=[]
initial_entropies=[]
rmses=[]

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
    #next_x_tuple=tuple([float(i) for i in next_x]) #converting composition into tuple.
    next_y=true_y_dict[float(next_x)]
    #updating model with composition and y-value
    dataset = dataset + Dataset(X=jnp.atleast_2d(next_x), y=jnp.atleast_2d(next_y))
    designs = jnp.delete(designs, (designs == next_x).argmax())[:, jnp.newaxis]
    pred_mean, pred_cov, posterior, params = update_model(dataset, design_space, rng_key, update_params=False)


df=pd.DataFrame()
df['means']=means
df['stds']=stds
df['iteration']=[i for i in range(iterations)]
df.to_csv(f'../data/chaase_meanfield_{num_y}_{num_samples}.csv')

'''
###Fourth fig
fig = plt.figure()
ax = fig.add_subplot(111)
size=14
rcParams['font.family'] = 'Arial'
norm = cm.colors.Normalize(vmax=1, vmin=0)
cmap = plt.cm.get_cmap('Blues')
colors = cm.ScalarMappable(norm=norm, cmap=cmap)

num_samples=100
samples=sample_from_posterior(pred_mean,pred_cov,design_space,num_samples,envelopes=False)
hull_samples=[]
Ys=[]
for x in range(num_samples):
    Y=samples[:,x]
    lin_comb=get_lin_comb(pts,endpoint_indices,Y)
    Y_zeroed=Y-lin_comb
    Ys.append(Y_zeroed)
    E_hull=get_hull_energies_oneD(design_space,Y_zeroed,endpoint_indices=endpoint_indices)
    hull_samples.append(E_hull)

hull_samples=np.vstack(hull_samples)
diff_entropies=differential_entropy(hull_samples) #vector of length knot_N
diff_entropies=np.nan_to_num(diff_entropies,neginf=-3)

num_plot=100
for x in range(num_plot):
    plt.plot(knot_x,hull_samples[x],color='black',ls='-',linewidth=2,alpha=1/20)
    #plt.plot(knot_x,Ys[x],color=cmap(norm(x/3)),ls='-.',linewidth=2,alpha=1)
plt.plot(knot_x,diff_entropies, color='black', linewidth=2)

plt.tick_params(direction='in',which='both',labelsize = size)
plt.xlabel('Composition',size=size)
plt.ylabel('Energy',size=size)
ax.set_aspect(1.0/ax.get_data_ratio())
plt.savefig('../figs/10_3030_iteration.png')
#plt.show()

plt.figure()
plt.plot(knot_x,diff_entropies, color=cmap(norm(1)), linewidth=2)
plt.xlabel('Composition',size=size)
plt.ylabel('Entropy',size=size)

ax.set_aspect(1.0/ax.get_data_ratio())
plt.show()


#plt.savefig('../figs/hull_samples_0502.png',dpi=600)
'''
