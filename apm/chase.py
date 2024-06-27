import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import time

from .gp.kernels import Matern52
from .mcmc import slice_sample_hypers
from .hull import lower_hull_points

log = logging.getLogger()

class CHASE:
  def __init__(self, cfg, source, rng):
    self.cfg = cfg
    self.source = source
    self.rng = rng
    self.kernel_fn = globals()[cfg.gp.kernel]
    self.comps = []
    self.energies = []

    self._hyper_init()

  def _hyper_init(self):
    ''' Initialize the hyperparameters the first time. '''
    ls_rng, amp_rng, self.rng = jrnd.split(self.rng, 3)

    self.init_ls = jrnd.uniform(
        ls_rng, 
        (self.source.num_phases, self.source.num_species,),
        minval=self.cfg.gp.lengthscale_prior[0], 
        maxval=self.cfg.gp.lengthscale_prior[1],
      )
    log.info("Initial lengthscales: %s" % self.init_ls)
    self.init_amp = jrnd.uniform(
        amp_rng, 
        (self.source.num_phases,),
        minval=self.cfg.gp.amplitude_prior[0],
        maxval=self.cfg.gp.amplitude_prior[1],
      )
    log.info("Initial amplitudes: %s" % self.init_amp)

  def corner_init(self):
    ''' Initialize by evaluating the pure compositions. '''
    candidates = self.source.all_candidates

    # Get each pure composition.
    for ii in range(self.source.num_species):
      pure = jnp.zeros(self.source.num_species).at[ii].set(1.0)
      idx = jnp.where(
          jnp.all(candidates == pure[jnp.newaxis,:], axis=1),
          size=1,
          fill_value=-1,
        )[0]
      if idx == -1:
        raise ValueError("Could not find pure composition.")
      self.comps.append(idx)
      
      phase_energies = []
      for jj in range(self.source.num_phases):
        phase_energies.append(self.source.eval_phase(idx,jj))
      self.energies.append(jnp.array(phase_energies).ravel())
      log.info("Pure composition %d has energies %s." % (ii+1, self.energies[-1]))

  def select_next(self):
    ''' Select the next composition to evaluate. '''

    X = self.source.all_candidates[jnp.array(self.comps).ravel(),:]
    Y = jnp.array(self.energies)

    # Generate samples from the posterior on hyperparameters.
    hyper_rng, self.rng = jrnd.split(self.rng)
    t0 = time.time()
    ls_samples, amp_samples = slice_sample_hypers(
        hyper_rng,
        X,
        Y,
        self.cfg,
        self.init_ls,
        self.init_amp,
        self.cfg.chase.hypers.num_samples + self.cfg.chase.hypers.num_burnin,
        self.cfg.chase.hypers.num_thin,  
      )
    t1 = time.time()
    log.info("Hyperparameter sampling took %f seconds." % (t1-t0))
    
    # Use the last values for the initialization next time.
    self.init_ls = ls_samples[-1]
    self.init_amp = amp_samples[-1]

    # Chop off the burn-in samples.
    ls_samples = ls_samples[self.cfg.chase.hypers.num_burnin:]
    amp_samples = amp_samples[self.cfg.chase.hypers.num_burnin:]

    # 


    # # Compute the minimum energies across phases.
    # min_energies = jnp.min(posterior_samples, axis=2)

    # #Compute the convex hull mask of the minima for each posterior sample.
    # tight = jax.vmap(
    #   jax.vmap(
    #       lower_hull_points,
    #       in_axes=(None, 0),
    #     ),
    #     in_axes=(None, 0),
    #   )(self.source.all_candidates, min_energies)
    # tight.block_until_ready()
    # t4 = time.time()
    # log.info("Convex hull computation took %f seconds." % (t4-t3))
