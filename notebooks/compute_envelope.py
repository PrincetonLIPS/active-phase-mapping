import jax
import jax.numpy as jnp 
import jax.random as jrnd
import jax.scipy.stats as jsps
import jax.scipy.linalg as spla
from jax.config import config
config.update("jax_enable_x64", True)

#import gpjax as gpx
from jax import jit
#import jaxkern as jk
#import optax as ox
#from jaxutils import Dataset

#from elliptical_slice_sampler_jax import elliptical_slice_jax
#from gp_model import make_preds, update_model
#from search_no_gpjax import sample_from_posterior

# non-jax utilities
from scipy.spatial import ConvexHull
import numpy as np

"""
This module provides the ConvexHull class.
"""

import itertools
import numpy as np
from typing import List, Sized, Union
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull as ConvexHullSciPy
from scipy.spatial.qhull import QhullError


def convex_envelope(concentrations, energies):
    assert len(concentrations) == len(energies)
    # Prepare data in format suitable for SciPy-ConvexHull
    #concentrations = np.array(concentrations)
    #energies = np.array(energies)
    points = np.column_stack((concentrations, energies))
    dimensions = len(points[0]) - 1

    # Construct convex hull
    hull = ConvexHullSciPy(points, qhull_options='Qs')
    structures = hull.vertices

    concentrations = points[hull.vertices, :-1]
    energies = points[hull.vertices, -1]
    
    #concentrations = points[hull.vertices][:, 0:-1]
    if dimensions == 1:
        concentrations = concentrations.ravel()
        inds = concentrations.argsort()
        concentrations = concentrations[inds]
        energies = energies[inds]
        structures = structures[inds]

    # Remove points that are above the "pure components plane"
    if dimensions == 1:
        vertices = np.array([concentrations.argmin(), concentrations.argmax()])
    else:
        concentration_hull = ConvexHullSciPy(concentrations, qhull_options='Qs')
        vertices = concentration_hull.vertices
        
    concentrations, energies, structures = remove_points_above_tie_plane_vectorized(vertices, structures, concentrations, energies)

    return structures

def remove_points_above_tie_plane_vectorized(vertices, structures, concentrations, energies):
    
    tol = 5e-3
    dimensions = len(concentrations.shape)
    
    mask = np.ones(concentrations.shape[0], dtype=bool)
    mask[vertices] = False
    
    for plane in itertools.combinations(vertices, min(len(vertices), dimensions + 1)):
        
        plane_concentrations = concentrations[list(plane)]
        plane_energies = energies[list(plane)]
        
        plane_energy_pure = griddata(plane_concentrations, plane_energies, concentrations[mask], method='linear')
        mask[mask] &= plane_energy_pure >= energies[mask] - tol
    
    # include
    mask[vertices] = True
    
    return concentrations[mask], energies[mask], structures[mask]

    
    
#def convex_envelope(x, fs, tol):
    #data = {"concentration": x, "mixing_energy": fs}
    #hull = ConvexHullMod(data['concentration'], data['mixing_energy'])
        
    #if tol > 0:
    #    low_energy_structures = hull.extract_low_energy_structures(data['concentration'], data['mixing_energy'], energy_tolerance=tol)
    
#    return compute_hull(x, fs, tol=tol)[2]



# TODO: check this for correctness
def convelope(design_space, knot_y):
    """Computes the convex envelope"""

    N, D = design_space.shape
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO: 
    #deriv_marg_var = np.max(jnp.diag(d_kernel(knot_x, knot_x, ls)))
    deriv_marg_var = 100
    s = jnp.linspace(-3*jnp.sqrt(deriv_marg_var), 3*jnp.sqrt(deriv_marg_var), 200)
    ss = jnp.meshgrid(*[s.ravel()]*D)
    s = jnp.array([sx.flatten() for sx in ss]).T
    knot_y = jnp.atleast_2d(knot_y) # samples x num_primal
    prod = (design_space @ s.T).T
    
    # compute the conjugate
    lft1 = jnp.max(prod[jnp.newaxis,:,:] - knot_y[:,jnp.newaxis,:],  axis=2) # samples x num_dual
    # compute the biconjugate
    lft2 = jnp.max(prod[jnp.newaxis,:,:] - lft1[:,:,jnp.newaxis],  axis=1) # samples x num_primal
    
    return lft2


def is_vertex(points, tol):
    N, D = points.shape
    vertices = convex_envelope(points[:, :-1], points[:, -1])
    s = np.zeros(N)
    s[vertices] = 1
    return s.astype("bool")

@jit
def is_tight(design_space, true_y, tol):

    points = jnp.hstack([design_space, true_y[:, jnp.newaxis]])
    _scipy_hull = lambda points: is_vertex(points, tol) 

    result_shape_dtype = jax.ShapeDtypeStruct(
          shape=jnp.broadcast_shapes(true_y.shape),
          dtype='bool')

    return jax.pure_callback(_scipy_hull, result_shape_dtype, points, vectorized=False)