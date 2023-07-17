import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

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

n_grid, dimensions, iterations=(21, 3, 228)

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
num_tights = 2; tights_baseline = [num_tights]


###Determining true_hull
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

###Active search 
means=[]
stds=[]
initial_entropies=[]
for i in range(iterations):

    print("Iteration: ", i)
    #Calculating initial information entropy
    initial_entropy=calc_entropy(pts=pts,endpoint_indices=endpoint_indices,knot_N=knot_N,pred_mean=pred_mean,pred_cov=pred_cov,
    design_space=design_space,num_samples=100,get_avg_pred=False)
    initial_entropies.append(initial_entropy)

    # get candidate design
    #next_x, entropy_change = get_next_candidate_baseline(posterior, params, dataset, designs, design_space)
    next_x=fps_point(designs,dataset)
    #print(next_x)
    # get the index of candidate w.r.t. full design space
    next_x_ind = (design_space == next_x).sum(1).argmax()

    dataset = dataset + Dataset(X=jnp.atleast_2d(next_x), y=jnp.atleast_2d(true_y[next_x_ind]))
    designs = jnp.delete(jnp.array(designs), (jnp.array(designs) == next_x).sum(1).argmax(), axis=0)

    pred_mean, pred_cov, posterior, params = update_model(dataset, design_space, rng_key, update_params=False)

    ###4
    #characterizaing predictive capability
    final_entropy, avg_pred=calc_entropy(pts=pts,endpoint_indices=endpoint_indices,knot_N=knot_N,pred_mean=pred_mean,pred_cov=pred_cov,
    design_space=design_space,num_samples=100,get_avg_pred=True)

    errors=jnp.abs(true_hull-avg_pred)
    means.append(jnp.mean(errors))
    stds.append(jnp.std(errors))
    print(means)

df=pd.DataFrame()
df['means']=means
df['stds']=stds
df['Initial_Entropy']=initial_entropies
df['iteration']=[i for i in range(iterations)]
df.to_csv('../data/fps.csv')
