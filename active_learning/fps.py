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
from os import system

#JAX
import jax
import jax.numpy as jnp
import jax.random as jrnd
from jaxutils import Dataset

# Imports from our code base
from utils import *
from gp_model import update_model, make_preds
from generate_function import generate_true_function

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
args = parser.parse_args()

system(f'mkdir {args.directory}')
master_hull_dict={}
master_energy_dict={}
master_dataset_dict={}
master_problem_setup_dict={}
dfs=[]
seeds=np.array([ 0, 10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27, 30, 31, 35, 36, 37, 40, 41, 42, 45, 46, 47, 50, 60, 61, 62, 65, 66, 67, 70, 71, 72, 75, 76, 77, 80, 81, 82, 85, 86, 87])

for seed in seeds:
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
    iterations =int((knot_N-dimensions)*num_polymorphs)+1
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

    counter = 0

    for it in tqdm(range(iterations)):
        print("Iteration: ", it)
        initial_time=time()
        #finding min_energies
        observed_energies=[]
        lengths=[]
        for i in range(num_polymorphs):
            energies=np.array(poly_dict[i]['dataset'].y+poly_dict[i]['mean'])
            observed_energies.append(energies.ravel())
            lengths.append(len(energies))
        #if lengths don't match, pad with arbitrarily large number.
        for index,i in enumerate(observed_energies):
            if len(i)<np.max(lengths):
                observed_energies[index]=np.append(i,10) #arbitrarily high number
        min_observed_energies=np.min(observed_energies,axis=0)

        #finding lin_comb
        full_xs=[]
        xs=poly_dict[0]['dataset'].X
        for coord in xs:
            full_xs.append(elaborate_point(coord))
        parent_energies=min_observed_energies[0:dimensions]
        lin_comb=np.dot(np.array(full_xs),parent_energies)
        full_lin_comb=np.dot(pts,parent_energies)

        #making full-length observation vector to compare with hulls

        base=np.ones(len(design_space))*10
        for index,i in enumerate(design_space):
                if np.any(np.all(i==xs,axis=1)): #works
                    sub_y_val = min_observed_energies[np.all(i==xs,axis=1)][0]
                    if base[index]>sub_y_val:
                        base[index]=sub_y_val


        #Y_zeroed
        Y_zeroed=min_observed_energies-lin_comb
        #Calculating hull
        points=np.column_stack((poly_dict[0]['dataset'].X,Y_zeroed))[Y_zeroed<=0]
        if len(points)>dimensions:
            hull=ConvexHull(points)
            new_eqns=reconfigure_eqns_nd(hull.equations)

            ###2
            E_hull=[]
            for point in design_space:
                prospective_hull_energies=[]
                new_point=np.append(point,1) #appending 1 so dot product works.
                #2a
                for eq in new_eqns:
                    prospective_hull_energy=np.dot(new_point,eq) #Get energy E=<x,1> dot <-A/B,-C/B>
                    prospective_hull_energies.append(prospective_hull_energy)
                prospective_hull_energies=np.array(prospective_hull_energies)
                #2b
                mask1=prospective_hull_energies<-1e-5
                if np.sum(mask1)>0:
                    hull_energy=np.max(prospective_hull_energies[mask1])
                    E_hull.append(hull_energy)
                else:
                    E_hull.append(0)
            ###3
            #If endpoints actually have E of zero, then hull energy should be zero as well.
            for x in endpoint_indices:
                E_hull[x]=0
            E_hull=np.array(E_hull)+full_lin_comb
            vertices=base-E_hull<tol
            classifications=jnp.zeros(knot_N).at[vertices].set(1)
            print("success")

        else:
            E_hull=np.zeros(knot_N) + full_lin_comb
            classifications=jnp.zeros(knot_N).at[np.array(endpoint_indices)].set(1)


        #Quantifying error.
        #Energy samples are min_curves.
        energy_errors=np.abs(true_e_hull-E_hull)
        energy_error_means.append(np.mean(energy_errors))
        energy_error_stds.append(np.std(energy_errors))
        false_positive_rate, true_positive_rate = roc(pred_vertices=classifications, true_vertices=true_classifications)
        false_positive_rates.append(false_positive_rate)
        true_positive_rates.append(true_positive_rate)

        if it != iterations-1:
            next_x=fps_point(poly_dict[counter]['designs'],poly_dict[counter]['dataset'])
            print(next_x)

            next_x_ind = (design_space == next_x).sum(1).argmax()
            next_y=poly_dict[counter]['true_y'][next_x_ind] #unnormalized y

            #updating model with composition and y-value
            y_array_unnormalized=poly_dict[counter]['dataset'].y+poly_dict[counter]['mean']
            y_array_unnormalized=jnp.append(y_array_unnormalized, next_y)
            poly_dict[counter]['mean']=y_array_unnormalized.mean()
            normalized_y = (y_array_unnormalized - poly_dict[counter]['mean'])#(y_array - poly_dict[i]['mean'])/poly_dict[i]['std']
            x_data=jnp.vstack([poly_dict[counter]['dataset'].X,next_x])#[:, jnp.newaxis]

            poly_dict[counter]['dataset'] = Dataset(X=x_data, y=normalized_y[:, jnp.newaxis])
            poly_dict[counter]['pred_mean'], poly_dict[counter]['pred_cov'], poly_dict[counter]['posterior'], poly_dict[counter]['params'] = update_model(
            poly_dict[counter]['dataset'], design_space, rng_key, update_params=False)
            poly_dict[counter]['designs'] = jnp.delete(jnp.array(poly_dict[counter]['designs']), (jnp.array(poly_dict[counter]['designs']) == next_x).sum(1).argmax(), axis=0)

        if counter<num_polymorphs-1:
            counter+=1
        elif counter == num_polymorphs-1:
            counter = 0

        duration=time()-initial_time
        its.append(it)
        iteration_times.append(duration)
        #Saving results
        df=pd.DataFrame()

        df['energy_error_means']=energy_error_means
        df['energy_error_stds']=energy_error_stds
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

    master_df=pd.concat(dfs)
    master_df.to_csv(f'{args.directory}/performance.csv')

'''
for i in range(num_polymorphs):
    plt.plot(poly_dict[i]['dataset'].X,poly_dict[i]['dataset'].y+poly_dict[i]['mean'],'o')

plt.plot(poly_dict[0]['dataset'].X,min_observed_energies,'o',color='black')
plt.plot(poly_dict[0]['dataset'].X,lin_comb,'o',color='orange')
#plt.plot(design_space.ravel(),E_hull,'o',color='orange')
plt.show()
'''
