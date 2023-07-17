import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla
from jax.config import config
from jaxutils import Dataset

config.update("jax_enable_x64", True)
import numpy.random as npr
import numpy as np
from numpy import linalg
from scipy.spatial import ConvexHull
from itertools import product
import random
from copy import deepcopy

from gp_model import update_model, make_preds

###Coordinate functions
def nD_coordinates(dimensions,start,stop,steps):
    '''
    Produces list of compositions that fall within simplex.
    dimensions: number of dimensions that the simplex has -1 (e.g. triangle takes 3, line takes 2).
    start: starting coordinate -- usually 0
    stop: ending coordinate -- usually 1
    steps: number of steps going from start to stop.

    Example:
    nD_coordinates(3,0,1,11) returns all the coordinates for a ternary simplex,
    where one line of the triangle has 11 compositions.
    '''
    x=np.linspace(start,stop,steps)
    coords=product(x,repeat=dimensions)
    lst=[]
    for x in coords:
        if np.abs(np.sum(x)-1.0)<1e-5:
            x=np.round(x,decimals=5)
            lst.append(x)
    return lst

def get_endpoint_indices(dimensions,pts):
    '''
    Generalized approach for enumerating endpoint coordinates.
    dimensions: number of dimensions (int)
    pts: list of all compositions. Must have all dimensions--no implied dimension.
    returns list of endpoint indices.
    '''
    endpoints=[]
    for x in range(dimensions):
        base=np.zeros(dimensions)
        base[x]=1
        endpoints.append(base)
    #determining indices of endpoint coordinates
    endpoint_indices=[]#order matters!
    for x in endpoints:
        index=np.argmax((np.array(pts)==x).sum(1)) #determines index corresponding to endpoint.
        endpoint_indices.append(index)
    return endpoint_indices
###

###Hull functions
def reconfigure_eqns_nd(eqns):
    '''
    QHull returns a series of linear equations that define the convex hull.
    We use these equations to derive the energy of the hull for a given composition.
    Here we algebraically reformulate these equation such that when dotted with composition,
    the energy is produced.

    Params:
    eqns: equations produced from QHull.

    Example:
    hull=ConvexHull(points)
    new_eqns=reconfigure_eqns_nd(hull.equations)

    Algebra:
    Formula given by hull: Ax + By + ... +YE + Z = 0
    Rearranging: E=(-Ax -By -...-Z)/Y
    As a dot product: E=<x,y,...,1> dot <-A/Y,-B/Y,...-Z/Y>

    Steps:
    1) remove second to last coefficient (corresponding to Energy).
    2) Divide by second to last coefficient and multiply by -1.
    '''
    new_eqns=[]
    for x in eqns:
        if x[-2]!=0:
            new = -np.append(x[:-2],x[-1])/x[-2]
            new_eqns.append(new)
        else:
            pass
    return np.array(new_eqns)

def get_hull_energies(design_space,Y,endpoint_indices):
    '''
    1) Build hull from non-positive values.
    2) Get energy from derived hull across entire domain.
        2a) For a point, calculate its energy for all possible hull_simplices.
        2b) Remove any zeros (corresponding to wrapping from above).
    3) Add back in 0-values for endpoints.

    design_space: compositions
    Y: Energy
    '''
    ###1
    #Building hull from non-positive vals.
    points=np.column_stack((design_space,Y))[Y<=0]
    hull=ConvexHull(points)
    new_eqns=reconfigure_eqns_nd(hull.equations)

    ###2
    hull_energies=[]
    for point in design_space:
        prospective_hull_energies=[]
        new_point=np.append(point,1) #appending 1 so dot product works.
        #2a
        for eq in new_eqns:
            prospective_hull_energy=np.dot(new_point,eq) #Get energy E=<x,1> dot <-A/B,-C/B>
            prospective_hull_energies.append(prospective_hull_energy)
        prospective_hull_energies=np.array(prospective_hull_energies)
        #2b
        mask1=prospective_hull_energies<0
        hull_energy=np.max(prospective_hull_energies[mask1])
        hull_energies.append(hull_energy)

    ###3
    #If endpoints actually have E of zero, then hull energy should be zero as well.
    for x in endpoint_indices:
        hull_energies[x]=0

    return np.array(hull_energies)

'''
def convelope(design_space, knot_y):

    N, D = design_space.shape
    deriv_marg_var = 100
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO:
    #deriv_marg_var = np.max(jnp.diag(d_kernel(knot_x, knot_x, ls)))
    #print(deriv_marg_var)
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 500)
    ss = jnp.meshgrid(*[s.ravel()]*D)
    s = jnp.array([sx.flatten() for sx in ss]).T

    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    #print(knot_y.shape)

    prod = (design_space @ s.T).T
    #print(prod.shape)

    # compute the conjugate
    lft1 = jnp.max(prod[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    #print(lft1.shape)
    # compute the biconjugate
    lft2 = jnp.max(prod[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal

    return lft2
'''
###


def calc_entropy(pts=None,endpoint_indices=None,knot_N=None,pred_mean=None,pred_cov=None,design_space=None,num_samples=None,get_avg_pred=False):
    tol=1e-3
    samples=sample_from_posterior(pred_mean,pred_cov,design_space,num_samples,envelopes=False)
    avg_pred=jnp.zeros(knot_N) #vector of length knot_N
    for x in range(num_samples):
        Y=samples[:,x]
        lin_comb=get_lin_comb(pts,endpoint_indices,Y)
        Y_zeroed=Y-lin_comb
        try: #Some functions might have only positive Y_zeroed values. Can't build a hull out of those functions. In that case, just endpoints are on hull.
            E_above_hull=Y_zeroed-get_hull_energies(design_space,Y_zeroed,endpoint_indices=endpoint_indices)
            vertices=E_above_hull<tol
            classifications=jnp.zeros(knot_N).at[vertices].set(1)
        except:
            classifications=jnp.zeros(knot_N).at[np.array(endpoint_indices)].set(1)
        avg_pred+=classifications
    avg_pred=avg_pred/num_samples #vector of length knot_N
    entropy = -avg_pred*jnp.log(avg_pred) -(1-avg_pred)*jnp.log(1-avg_pred) #vector of length knot_N
    total_entropy = jnp.sum(jnp.nan_to_num(entropy)) #scalar -- entropy of whole system.

    if get_avg_pred:
        return total_entropy,avg_pred
    else:
        return total_entropy



def get_lin_comb(pts,endpoint_indices,Y):
    '''
    Gets the linear combination of the endpoint energies for the entire space.
    Subtracting off the lin_comb from Y gets you a function that is zeroed at the endpoints.
    This is critical for calculating the hull energies.
    '''
    endpoint_energies=np.array(Y)[endpoint_indices]
    lin_comb=np.dot(pts,endpoint_energies) #Gives the linear combination of all three vertices.
    return lin_comb


def sample_from_posterior(pred_mean, pred_cov, design_space, T,envelopes=False):
    # TODO: clean this up -- not necessary
    # sample functions ~ posterior
    N_designs = design_space.shape[0]
    pred_cK = spla.cholesky(pred_cov)
    # get T samples from the posterior
    pred_Y = pred_cK.T @ npr.randn(N_designs, T) + pred_mean[:, jnp.newaxis]
    # get s by computing the vector of tights w.r.t. posterior samples
    if envelopes:
        envelopes = convelope(design_space, pred_Y.T)
        return pred_Y, envelopes
    else:
        return pred_Y

def elaborate_point(reduced_point):
    tmp_array=np.array(reduced_point)
    return np.append(tmp_array,1-np.sum(tmp_array))

###Sampling Methods
def EIG_chaase_meanfield(composition, dataset, seed, index_dict, pred_mean, pred_cov,
    design_space, num_y, initial_entropy, pts, endpoint_indices, knot_N, num_samples):
    temp_comp_tuple=tuple([float(i) for i in composition])
    index=index_dict[temp_comp_tuple]
    entropies=[] #list of entropies for various y samples.
    #Loop calculates entropy for each possible y.
    samples=sample_from_posterior(pred_mean,pred_cov,design_space,num_y,envelopes=False) #samples y curves.
    for x in range(num_y): #inner loop, ys.
        function=samples[:,x] #x-th curve.
        y_0=function[index] #Get y-value for given curve
        #Make copy of dataset and add y-value.
        tmp_dataset=deepcopy(dataset)
        tmp_dataset = tmp_dataset + Dataset(X=jnp.atleast_2d(composition), y=jnp.atleast_2d(y_0)) #updating dataset with new value
        #Update the model given new y-value.
        npr.seed(seed); rng_key = jrnd.PRNGKey(seed)
        pred_mean, pred_cov, posterior, params = update_model(tmp_dataset, design_space, rng_key, update_params=False) #update the model.
        #Calculate the resulting entropy
        entropy=calc_entropy(pts=pts,endpoint_indices=endpoint_indices,knot_N=knot_N,pred_mean=pred_mean,pred_cov=pred_cov,
        design_space=design_space,num_samples=num_samples,get_avg_pred=False)
        #Save to list of entropies (containing one entry for each sampled y)
        entropies.append(entropy)

    entropies=jnp.array(entropies)
    expected_final_entropy=jnp.mean(entropies) #expectation value of entropies.
    EIG=initial_entropy-expected_final_entropy

    return EIG

def fps_point(designs,dataset):
    obs_pts = []
    for x in dataset.X:
        obs_pts.append(elaborate_point(x))
    obs_pts=np.array(obs_pts)

    min_dists=[]
    for point in designs:
        full_pt=elaborate_point(point)
        distances = linalg.norm(full_pt-obs_pts,axis=1)
        min_distance=np.min(distances)
        min_dists.append(min_distance)
    index_fps_point=np.argmax(min_dists)
    fps_point=designs[index_fps_point]
    return fps_point


def random_sample(design_space):
    return random.choice(design_space)
###

'''
def classify(knot_N,knot_x,Y):
    points=np.column_stack((knot_x,Y))
    hull=ConvexHull(points)
    true_vertices=hull.vertices[hull.points[hull.vertices][:,1]<0]
    classifications=jnp.zeros(knot_N).at[true_vertices].set(1) #rewrite in np?
    classifications=classifications.at[jnp.array([0,-1])].set(1) #rewrite in np?
    return classifications
'''
