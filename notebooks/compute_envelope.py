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


def convex_envelope_old(concentrations, energies):
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

    


# TODO: check this for correctness
def convelope(design_space, knot_y):
    """Computes the convex envelope"""

    N, D = design_space.shape
    d_kernel = jax.jit(jax.vmap(jax.grad(jax.grad(lambda x1, x2, ls: kernel_old(x1, x2, ls)[0,0], argnums=0), argnums=1), in_axes=(0,0,None)))
    # TODO: 
    #deriv_marg_var = np.max(jnp.diag(d_kernel(design_space, design_space, ls)))
    deriv_marg_var = 200
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

def convex_envelope(x, fs):
    """
    Get vertices on the lower convex envelope.
    """
    points = np.column_stack((x, fs))
    hull = ConvexHull(points)
    down_facing = hull.equations[:,-2] < 0
    return np.unique(hull.simplices[down_facing,:].ravel())

def is_vertex(points):
    """
    Computes a binary vector of whether or not each vertex is tight
    """
    N, D = points.shape
    vertices = convex_envelope(points[:, :-1], points[:, -1])
    s = np.zeros(N)
    s[vertices] = 1
    return s.astype("bool")

@jit
def is_tight(design_space, true_y):
    """
    Function called by the likelihood evaluation in slice sampling.
    """

    points = jnp.hstack([design_space, true_y[:, jnp.newaxis]])
    _scipy_hull = lambda points: is_vertex(points) 

    result_shape_dtype = jax.ShapeDtypeStruct(
          shape=jnp.broadcast_shapes(true_y.shape),
          dtype='bool')

    return jax.pure_callback(_scipy_hull, result_shape_dtype, points, vectorized=False)








"""
This module provides the ConvexHull class.
"""

import itertools
import numpy as np
from typing import List, Sized, Union
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull as ConvexHullSciPy
from scipy.spatial import QhullError

class ConvexHullMod:
    """This class provides functionality for extracting the convex hull
    of the (free) energy of mixing. It is based on the `convex hull
    calculator in SciPy
    <http://docs.scipy.org/doc/scipy-dev/reference/\
generated/scipy.spatial.ConvexHull.html>`_.
    Parameters
    ----------
    concentrations : list(float) or list(list(float))
        concentrations for each structure listed as ``[[c1, c2], [c1, c2],
        ...]``; for binaries, in which case there is only one independent
        concentration, the format ``[c1, c2, c3, ...]`` works as well.
    energies : list(float)
        energy (or energy of mixing) for each structure
    Attributes
    ----------
    concentrations : np.ndarray
        concentrations of the `N` structures on the convex hull
    energies : np.ndarray
        energies of the `N` structures on the convex hull
    dimensions : int
        number of independent concentrations needed to specify a point in
        concentration space (1 for binaries, 2 for ternaries etc.)
    structures : list(int)
        indices of structures that constitute the convex hull (indices are
        defined by the order of their concentrations and energies are fed when
        initializing the ConvexHull object)
    Examples
    --------
    A `ConvexHull` object is easily initialized by providing lists of
    concentrations and energies::
        >>> data = {'concentration': [0,    0.2,  0.2,  0.3,  0.4,  0.5,  0.8,  1.0],
        ...         'mixing_energy': [0.1, -0.2, -0.1, -0.2,  0.2, -0.4, -0.2, -0.1]}
        >>> hull = ConvexHull(data['concentration'], data['mixing_energy'])
    Now one can for example access the points along the convex hull directly::
        >>> for c, e in zip(hull.concentrations, hull.energies):
        ...     print(c, e)
        0.0 0.1
        0.2 -0.2
        0.5 -0.4
        1.0 -0.1
    or plot the convex hull along with the original data using e.g., matplotlib::
        >>> import matplotlib.pyplot as plt
        >>> plt.scatter(data['concentration'], data['mixing_energy'], color='darkred')
        >>> plt.plot(hull.concentrations, hull.energies)
        >>> plt.show(block=False)
    It is also possible to extract structures at or close to the convex hull::
        >>> low_energy_structures = hull.extract_low_energy_structures(
        ...     data['concentration'], data['mixing_energy'],
        ...     energy_tolerance=0.005)
    A complete example can be found in the :ref:`basic tutorial
    <tutorial_enumerate_structures>`.
    """

    def __init__(self,
                 concentrations: Union[List[float], List[List[float]]],
                 energies: List[float]) -> None:
        assert len(concentrations) == len(energies)
        # Prepare data in format suitable for SciPy-ConvexHull
        concentrations = np.array(concentrations)
        energies = np.array(energies)
        points = np.column_stack((concentrations, energies))
        self.dimensions = len(points[0]) - 1

        # Construct convex hull
        hull = ConvexHullSciPy(points, qhull_options='Qs')

        # Collect convex hull points in handy arrays
        concentrations = []  # type: ignore
        energies = []  # type: ignore
        
        for vertex in hull.vertices:
            if self.dimensions == 1:
                concentrations.append(points[vertex][0])
            else:
                concentrations.append(points[vertex][0:-1])
            energies.append(points[vertex][-1])
        
        concentrations = np.array(concentrations)
        energies = np.array(energies)

        structures = hull.vertices
        # If there is just one independent concentration, we'd better sort
        # according to it
        if self.dimensions == 1:
            ces = list(zip(*sorted(zip(concentrations, energies, structures))))
            self.concentrations = np.array(ces[0])
            self.energies = np.array(ces[1])
            self.structures = np.array(ces[2])
        else:
            self.concentrations = concentrations
            self.energies = energies
            self.structures = structures

        # Remove points that are above the "pure components plane"
        self._remove_points_above_tie_plane()

    def _remove_points_above_tie_plane(self, tol: float = 1e-3) -> None:
        """
        Remove all points on the convex hull that correspond to maximum rather
        than minimum energy.
        Parameters
        ----------
        tol
            Tolerance for what energy constitutes a lower one.
        """

        # Identify the "complex concentration hull", i.e. the extremal
        # concentrations. In the simplest case, these should simply be the
        # pure components.
        if self.dimensions == 1:
            # Then the ConvexHullScipy function doesn't work, so we just pick
            # the indices of the lowest and highest concentrations.
            vertices = []
            vertices.append(np.argmin(self.concentrations))
            vertices.append(np.argmax(self.concentrations))
            vertices = np.array(vertices)
        else:
            concentration_hull = ConvexHullSciPy(self.concentrations,qhull_options='Qs')
            vertices = concentration_hull.vertices

        # Remove all points of the convex energy hull that have an energy that
        # is higher than what would be gotten with pure components at the same
        # concentration. These points are mathematically on the convex hull,
        # but in the physically uninteresting upper part, i.e. they maximize
        # rather than minimize energy.
        to_delete = []
        for i, concentration in enumerate(self.concentrations):
            # The points on the convex concentration hull should always be
            # included, so skip them.
            if i in vertices:
                continue

            # The energy obtained as a linear combination of concentrations on
            # the convex hull is the "z coordinate" of the position on a
            # (hyper)plane in the (number of independent concentrations +
            # 1)-dimensional (N-D) space. This plane is spanned by N points.
            # If there are more vertices on the convex hull, we need to loop
            # over all combinations of N vertices.
            for plane in itertools.combinations(vertices,
                                                min(len(vertices),
                                                    self.dimensions + 1)):
                # Calculate energy that would be gotten with pure components
                # with ascribed concentration.
                energy_pure = griddata(self.concentrations[np.array(plane)],
                                       self.energies[np.array(plane)],
                                       concentration,
                                       method='linear')

                # Prepare to delete if the energy was lowered. `griddata` gives
                # NaN if the concentration is outside the triangle formed by
                # the three vertices. The result of the below comparison is
                # then False, which is what we want.
                if energy_pure < self.energies[i] - tol:
                    to_delete.append(i)
                    break

        # Finally remove all points
        self.concentrations = np.delete(self.concentrations, to_delete, 0)
        self.energies = np.delete(self.energies, to_delete, 0)
        self.structures = list(np.delete(self.structures, to_delete, 0))

    def get_energy_at_convex_hull(self, target_concentrations:
                                  Union[List[float],
                                        List[List[float]]]) -> np.ndarray:
        """Returns the energy of the convex hull at specified concentrations.
        If any concentration is outside the allowed range, NaN is
        returned.
        Parameters
        ----------
        target_concentrations
            concentrations at target points
            If there is one independent concentration, a list of
            floats is sufficient. Otherwise, the concentrations ought
            to be provided as a list of lists, such as ``[[0.1, 0.2],
            [0.3, 0.1], ...]``.
        """
        if self.dimensions > 1 and isinstance(target_concentrations[0], Sized):
            assert len(target_concentrations[0]) == self.dimensions

        # Loop over all complexes of N+1 points to make sure that the lowest
        # energy plane is used in the end. This is needed in two dimensions
        # but in higher.
        hull_candidate_energies = []
        for plane in itertools.combinations(range(len(self.energies)),
                                            min(len(self.energies),
                                                self.dimensions + 1)):
            try:
                plane_energies = griddata(self.concentrations[list(plane)],
                                          self.energies[list(plane)],
                                          np.array(target_concentrations),
                                          method='linear')
            except QhullError:
                # If the points lie on a line, the convex hull will fail, but
                # we do not need to care about these "planes" anyway
                continue
            hull_candidate_energies.append(plane_energies)

        # Pick out the lowest energies found
        hull_energies = np.nanmin(hull_candidate_energies, axis=0)
        return hull_energies

    def extract_low_energy_structures(self, concentrations:
                                      Union[List[float],
                                            List[List[float]]],
                                      energies: List[float],
                                      energy_tolerance: float) -> List[int]:
        """Returns the indices of energies that lie within a certain
        tolerance of the convex hull.
        Parameters
        ----------
        concentrations
            concentrations of candidate structures
            If there is one independent concentration, a list of
            floats is sufficient. Otherwise, the concentrations must
            be provided as a list of lists, such as ``[[0.1, 0.2],
            [0.3, 0.1], ...]``.
        energies
            energies of candidate structures
        energy_tolerance
            include structures with an energy that is at most this far
            from the convex hull
        """
        # Convert to numpy arrays, can be necessary if, for example,
        # they are Pandas Series with "gaps"
        concentrations = np.array(concentrations)
        energies = np.array(energies)

        n_points = len(concentrations)
        if len(energies) != n_points:
            raise ValueError('concentrations and energies must have '
                             'the same length')

        # Calculate energy at convex hull for specified concentrations
        hull_energies = self.get_energy_at_convex_hull(concentrations)

        # Extract those that are close enough
        close_to_hull = [i for i in range(n_points)
                         if energies[i] <= hull_energies[i] + energy_tolerance]

        return close_to_hull
    
    
    
def convex_envelope_original(x, fs):
    data = {"concentration": x, "mixing_energy": fs}
    hull = ConvexHullMod(data['concentration'], data['mixing_energy'])
    return np.array(hull.structures)

def get_hull(x, fs):
    data = {"concentration": x, "mixing_energy": fs}
    hull = ConvexHullMod(data['concentration'], data['mixing_energy'])

    return np.array(hull.get_energy_at_convex_hull(data['concentration']))

