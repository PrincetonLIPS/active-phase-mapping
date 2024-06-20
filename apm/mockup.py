import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging

from .kernels import sqdist, Matern52

log = logging.getLogger()

class Mockup:
  ''' This construct a mockup energy function for testing purposes. '''

  def __init__(self, cfg):
    self.num_atoms   = cfg.mockup.num_atoms
    self.num_species = cfg.mockup.num_species
    self.num_phases  = cfg.mockup.num_phases
    self.noise       = cfg.mockup.noise
    self.seed        = cfg.mockup.seed
    self.ls_prior    = cfg.gp.lengthscale_prior
    self.amp_prior   = cfg.gp.amplitude_prior
    self.kernel      = cfg.gp.kernel
    self.jitter      = cfg.gp.jitter

    self.all_candidates = self._gen_candidates()
    self.energies       = self._gen_energies()

    log.info("Creating mockup with %d atoms, %d species, %d phases, and noise %f." \
      % (self.num_atoms, self.num_species, self.num_phases, self.noise))

  def _gen_candidates(self):
    ''' Compute all possible compositions, assuming a discrete number of atoms
    in the supercell can be simulated. '''
    cand = jnp.stack(
      jnp.meshgrid(*[jnp.arange(self.num_atoms)]*self.num_species), 
      axis=-1,
    ).reshape(-1, self.num_species)
    return cand[jnp.sum(cand, axis=1)==self.num_atoms,:]/self.num_atoms
  
  def _gen_energies(self):
    ''' Generate random energy functions. '''

    # Choose lengthscales and amplitudes from the prior.
    rng = jrnd.PRNGKey(self.seed)
    ls_rng, amp_rng, rng = jrnd.split(rng, 3)
    self.lengthscales = jrnd.uniform(
      ls_rng, 
      (self.num_phases, self.num_species),
      minval=self.ls_prior[0], 
      maxval=self.ls_prior[1],
    )
    self.amplitudes = jrnd.uniform(
      amp_rng, 
      (self.num_phases,),
      minval=self.amp_prior[0],
      maxval=self.amp_prior[1],
    )
    log.info("Lengthscales: %s" % self.lengthscales)
    log.info("Amplitudes: %s" % self.amplitudes)

    # Get the kernel function from the string in the config file.
    kernel_fn = globals()[self.kernel]

    # Construct the covariances.
    # We handle lengthscale, noise, and amplitude outside the kernel.
    # We vmap over the phases.
    apply_K = lambda X, ls, amp: amp * kernel_fn(X/ls) \
      + jnp.eye(X.shape[0]) * (self.noise+self.jitter)
    K = jax.vmap(apply_K, in_axes=(None, 0, 0))(
      self.all_candidates, 
      self.lengthscales, 
      self.amplitudes,
    )

    # Compute the Cholesky decompositions.
    chol_K = jax.vmap(jax.scipy.linalg.cholesky)(K)

    # Generate random energies.
    Z = jrnd.normal(rng, (self.num_phases, self.all_candidates.shape[0]))
    energies = jnp.einsum('ijk,ij->ik', chol_K, Z)

    return energies