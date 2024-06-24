import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging

from .utils import stars_and_bars
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
    self.num_rff     = cfg.gp.num_rff

    self.all_candidates = self._gen_candidates()
    self.energies       = self._gen_energies()

    log.info("Creating mockup with %d atoms, %d species, %d phases, and noise %f." \
      % (self.num_atoms, self.num_species, self.num_phases, self.noise))

  def _gen_candidates(self):
    ''' Compute all possible compositions, assuming a discrete number of atoms
    in the supercell can be simulated. '''
    return stars_and_bars(self.num_atoms, self.num_species).astype(jnp.float32)/self.num_atoms
  
  def _gen_energies(self):
    ''' Generate random energy functions. '''

    # Choose lengthscales and amplitudes from the prior.
    rng = jrnd.PRNGKey(self.seed)
    ls_rng, amp_rng, proj_rng, phase_rng, wts_rng = jrnd.split(rng, 3)
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
    # kernel_fn = globals()[self.kernel]

    # Generate random Fourier features.


    return energies
  
  def eval_phase(self, cand_idx, phase_idx):
    ''' Evaluate the energies. '''
    return self.energies[cand_idx, phase_idx]