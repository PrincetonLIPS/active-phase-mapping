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
from drew_nd import *

def plot_search(knot_x, poly_dict, design_space, num_samples, num_polymorphs, num_curves, pts, endpoint_indices, iteration, true_e_hull):
    color_list=['blue','purple','orange','red']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_samples=multi_GP_sample(poly_dict,design_space,num_samples)
    for i in range(num_polymorphs):
        plt.plot(np.ravel(poly_dict[i]['dataset'].X),np.ravel(poly_dict[i]['dataset'].y),'o',color='black')
        for x in range(num_samples):
            y=all_samples[i][:,x]
            plt.plot(knot_x,y,color=color_list[i],alpha=0.1,linewidth=1)


    min_curves=sample_min_curves(all_samples,poly_dict,num_curves,num_samples)
    avg_e_hull=np.zeros(len(knot_x))
    hull_samples=[]
    for x in range(num_samples):
        Y=min_curves[:,x]
        lin_comb=get_lin_comb(pts,endpoint_indices,Y)
        Y_zeroed=Y-lin_comb
        try:
            E_hull=get_hull_energies_oneD(design_space,Y_zeroed,endpoint_indices=endpoint_indices)
        except:
            E_hull=np.zeros(len(knot_x))
        final_E_hull=E_hull+lin_comb
        avg_e_hull+=final_E_hull

        hull_samples.append(final_E_hull)

    hull_samples=np.vstack(hull_samples)
    diff_entropies=differential_entropy(hull_samples) #vector of length knot_N
    diff_entropies=np.nan_to_num(diff_entropies,neginf=0)
    size=14
    rcParams['font.family'] = 'Arial'

    avg_e_hull=avg_e_hull/num_samples
    plt.plot(knot_x,avg_e_hull,color='grey',ls='--',alpha=1,linewidth=2)
    plt.plot(knot_x,true_e_hull,color='grey',alpha=1,linewidth=2)
    plt.tick_params(direction='in',which='both',labelsize = size)
    plt.xlabel('Composition',size=size)
    plt.ylabel('Energy',size=size)
    plt.ylim(-3,3)
    ax.set_aspect(1.0/ax.get_data_ratio())
    plt.title(f'Iteration #{iteration}')
   #plt.plot(knot_x,diff_entropies,color='black')

    plt.savefig(f'../figs/{iteration}_{num_polymorphs}.png',dpi=600)
