import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
##import ternary
from matplotlib import rcParams
import matplotlib.cm as cm
from glob import iglob
from time import time
import argparse
from copy import deepcopy
import seaborn as sns
from mpi4py import MPI
from os import system

#JAX
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jaxutils import Dataset

# Imports from our code base
from utils import *
from gp_model import update_model, make_preds
from base_policy import get_next_y, get_next_candidate_baseline
from generate_function import generate_true_function
from mpi import do_parallel

parser = argparse.ArgumentParser(description="Search parameters.")
parser.add_argument("-num_y", type=int, help="Number of y-points sampled per composition")
parser.add_argument("-num_samples", type=int, help="Number of samples per entropy calculation")
parser.add_argument("-num_curves", type=int, help="Number of combinations of samples per entropy calculation")
parser.add_argument('-entropy_type', type=str, help='Type of entropy calculation')
parser.add_argument('-cores', type=int, help='number of cores')
parser.add_argument('-nodes', type=int,help='number of nodes')
parser.add_argument('-num_comps', type=int, help='number of compositions considered for next design')
parser.add_argument('-directory', type=str, help='path to results directory')
parser.add_argument('-method', type=str, help='Type of Baseline search method')
parser.add_argument('-seed_range', type=int, nargs="+", help='Seed Range')
args = parser.parse_args()

system(f'mkdir {args.directory}')
master_hull_dict={}
master_energy_dict={}
master_dataset_dict={}
master_problem_setup_dict={}
dfs=[]

for seed in range(args.seed_range[0],args.seed_range[1]):
    #set seed
    npr.seed(seed); rng_key = jrnd.PRNGKey(seed)
    #Search space
    n_grid, dimensions, num_polymorphs = (11, 3, 3)
    num_curves=args.num_curves
    num_samples=args.num_samples
    num_y = args.num_y
    entropy_type = args.entropy_type
    method=args.method

    poly_dict={}
    for i in range(num_polymorphs):
        poly_dict[i]=deepcopy({'true_y':[],'design_space':[],'pred_mean':[],
        'pred_cov':[],'posterior':[],'params':[],'train_x':[],'train_y':[],
        'dataset':[]})

    #Producing grid
    pts=nD_coordinates(dimensions,0,1,n_grid)
    design_space=np.array(pts)[:,:dimensions-1]
    endpoint_indices = get_endpoint_indices(dimensions,pts)
    #removing endpoint indices from list of candidates.
    designs = [x for index,x in enumerate(design_space) if index not in endpoint_indices]
    knot_N = len(design_space)
    iterations = 150
    print(knot_N)
    alpha=[0,0.25,0.5]
    for i in range(num_polymorphs):
        #Generating energy  surfaces
        poly_dict[i]['true_y'] = generate_true_function(design_space, knot_N) + alpha[i]*i
        #poly_dict[i]['std'], poly_dict[i]['mean'] = poly_dict[i]['true_y'].std(), poly_dict[i]['true_y'].mean()
        #plt.plot(np.ravel(design_space),poly_dict[i]['true_y'])
        #plt.show()
        poly_dict[i]['true_y_dict'] = {}
        for comp, y in zip(design_space, poly_dict[i]['true_y']):
            comp=tuple([float(i) for i in comp])
            poly_dict[i]['true_y_dict'][comp]=y

        #Training on endpoint indices.
        x_lst,y_lst=[],[]
        for endpoint_index in endpoint_indices:
            x_lst.append(design_space[endpoint_index])
            y_lst.append(poly_dict[i]['true_y'][endpoint_index])
        y_array=jnp.array(y_lst)[:,jnp.newaxis]
        y_mean=jnp.array(y_lst)[:,jnp.newaxis].mean()
        y_std=jnp.array(y_lst)[:,jnp.newaxis].std()
        normalized_y =  y_array-y_mean #y_array - poly_dict[i]['mean']
        poly_dict[i]['std'] = deepcopy(y_std)
        poly_dict[i]['mean'] = deepcopy(y_mean)
        poly_dict[i]['dataset'] = deepcopy(Dataset(X=jnp.array(x_lst), y=normalized_y))
        poly_dict[i]['designs'] = deepcopy(designs)
        poly_dict[i]['pred_mean'], poly_dict[i]['pred_cov'], poly_dict[i]['posterior'], poly_dict[i]['params'] = update_model(
        poly_dict[i]['dataset'], design_space, rng_key, update_params=False) # Update the model given the data above
    index_dict={}
    for index,x in enumerate(design_space):
        if dimensions>2:
            comp_tuple=tuple([float(i) for i in x])
            index_dict[comp_tuple]=index
        else:
            index_dict[float(x)]=index #for 1D
    ###Determining true_hull
    #Producing min_curve
    all_true_curves=[]
    for i in range(num_polymorphs):
        all_true_curves.append(poly_dict[i]['true_y'])
    min_curve=np.min(np.vstack(all_true_curves),axis=0)
    #print(min_curve)
    #Set endpoints to zero.
    lin_comb=get_lin_comb(pts,endpoint_indices,min_curve)
    Y_zeroed=min_curve-lin_comb

    #Determining vertices of hull
    true_e_hull=lin_comb + get_hull_energies(design_space,Y_zeroed,endpoint_indices=endpoint_indices, dimensions=dimensions)
    tol=1e-3
    vertices=(min_curve-true_e_hull)<tol
    true_classifications=jnp.zeros(knot_N).at[vertices].set(1)
    print('ON HULL',np.sum(true_classifications))
    ###Active search
    energies={}
    scale=n_grid-1
    its=[]
    energy_error_means=[]
    energy_error_stds=[]
    false_positive_rates=[]
    true_positive_rates=[]
    initial_entropies=[]
    iteration_times=[]
    hull_dict={}
    energy_dict={}
    dataset_dict={}
    num_curves=args.num_curves
    num_samples=args.num_samples
    num_y = args.num_y

    problem_setup_dict={}
    problem_setup_dict['Grid Spacing']=n_grid
    problem_setup_dict['Number of Samples']=num_samples
    problem_setup_dict['Method']=method
    problem_setup_dict['Dimensions (Including Implicit']=dimensions
    problem_setup_dict['offset']=alpha
    problem_setup_dict['Number of Polymorphs']=1
    problem_setup_dict['Points']=pts
    problem_setup_dict['Design Space']=design_space
    problem_setup_dict['Lengthscale']=0.2
    problem_setup_dict['Kernel']='RBF'
    problem_setup_dict['Noise']=0
    problem_setup_dict['Variance']=1
    problem_setup_dict['Poly Dict']=poly_dict
    problem_setup_dict['True E_Hull']=true_e_hull
    problem_setup_dict['True Hull Vertices']=vertices
    problem_setup_dict['True Hull Classifications']=true_classifications
    problem_setup_dict['Energy Classification Tolerance']=tol
    problem_setup_dict['Index Dict']=index_dict
    master_problem_setup_dict[seed]=problem_setup_dict


    for it in tqdm(range(iterations)):
        print("Iteration: ", it)
        initial_time=time()

        #Quantifying error.
        #Energy samples are min_curves.
        hull_expected_energy, false_positive_rate, true_positive_rate, initial_entropy, hull_samples, energy_samples, all_poly_samples  =calc_expected_energy_or_entropy(
        design_space=design_space, num_curves=num_curves, num_samples=num_samples, knot_N=knot_N, pts=pts, endpoint_indices=endpoint_indices,
        dimensions=dimensions, true_hull_classifications=true_classifications, poly_dict=poly_dict, multi=True, entropy_type=entropy_type)

        hull_dict[it]=hull_samples
        energy_dict[it]=all_poly_samples
        dataset_dict[it]=deepcopy(poly_dict)
        energy_errors=np.abs(true_e_hull-hull_expected_energy)
        energy_error_means.append(np.mean(energy_errors))
        energy_error_stds.append(np.std(energy_errors))
        initial_entropies.append(initial_entropy)
        false_positive_rates.append(false_positive_rate)
        true_positive_rates.append(true_positive_rate)

        #Calculate EIG for each polymorph and over all compositions.
        data_dict = {} #dictionary mapping EIG to composition
        for i in range(num_polymorphs):

            #data_dict[i]
            dict_single_poly={}
            dict_list=do_parallel(
            designs=poly_dict[i]['designs'], dataset=poly_dict[i]['dataset'], seed=seed, index_dict=index_dict,
            pred_mean=poly_dict[i]['pred_mean'], pred_cov=poly_dict[i]['pred_cov'],
            design_space=design_space, num_y=num_y, initial_entropy=initial_entropy, pts=pts, endpoint_indices=endpoint_indices,
            knot_N=knot_N, num_curves=num_curves, num_samples=num_samples, poly_dict=poly_dict, polymorph_index=i, num_polymorphs=num_polymorphs,
            dimensions=dimensions, true_classifications=true_classifications, entropy_type=entropy_type)
            for x in dict_list:
                dict_single_poly.update(x)
            data_dict[i]=dict_single_poly


        #choosing next_x
        maxs={}
        for i in range(num_polymorphs):
            maxs[np.max(list(data_dict[i].keys()))]=i
        max_EIG=np.max(list(maxs.keys()))
        max_polymorph=maxs[max_EIG]
        next_x=data_dict[max_polymorph][max_EIG]
        next_x_ind = (design_space == next_x).sum(1).argmax()
        next_y=poly_dict[max_polymorph]['true_y'][next_x_ind] #unnormalized y


        #updating model with composition and y-value
        y_array_unnormalized=poly_dict[max_polymorph]['dataset'].y+poly_dict[max_polymorph]['mean']
        y_array_unnormalized=jnp.append(y_array_unnormalized, next_y)
        poly_dict[max_polymorph]['mean']=y_array_unnormalized.mean()
        normalized_y = (y_array_unnormalized - poly_dict[max_polymorph]['mean'])#(y_array - poly_dict[i]['mean'])/poly_dict[i]['std']
        x_data=jnp.vstack([poly_dict[max_polymorph]['dataset'].X,next_x])#[:, jnp.newaxis]
        poly_dict[max_polymorph]['dataset'] = Dataset(X=x_data, y=normalized_y[:, jnp.newaxis])
        poly_dict[max_polymorph]['pred_mean'], poly_dict[max_polymorph]['pred_cov'], poly_dict[max_polymorph]['posterior'], poly_dict[max_polymorph]['params'] = update_model(poly_dict[max_polymorph]['dataset'], design_space, rng_key, update_params=False)
        poly_dict[max_polymorph]['designs'] = jnp.delete(jnp.array(poly_dict[max_polymorph]['designs']), (jnp.array(poly_dict[max_polymorph]['designs']) == next_x).sum(1).argmax(), axis=0)

        duration=time()-initial_time
        its.append(it)
        iteration_times.append(duration)
        #Saving results
        df=pd.DataFrame()

        df['energy_error_means']=energy_error_means
        df['energy_error_stds']=energy_error_stds
        df['Entropy']=initial_entropies
        df['true_positive_rate']=true_positive_rates
        df['false_positive_rate']=false_positive_rates
        df['time (s)']=iteration_times
        df['iteration']=its
        df['cores']=args.cores
        df['nodes']=args.nodes
        df['Seed']=seed

        df.to_csv(f'{args.directory}/performance_seed_{seed}.csv')
        master_hull_dict[seed]=hull_dict
        master_energy_dict[seed]=energy_dict
        master_dataset_dict[seed]=dataset_dict
        if it == iterations-1:
            dfs.append(df)
        save_pickle(path=f'{args.directory}/hull.pkl',item=master_hull_dict)
        save_pickle(path=f'{args.directory}/energy.pkl',item=master_energy_dict)
        save_pickle(path=f'{args.directory}/observation.pkl',item=master_dataset_dict)
        save_pickle(path=f'{args.directory}/problem_setup.pkl',item=master_problem_setup_dict)

    master_df=pd.concat(dfs)
    master_df.to_csv(f'{args.directory}/performance.csv')
