import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import time

from .kernels import Matern52
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

    #X = self.source.all_candidates[jnp.array(self.comps).ravel(),:]
    #Y = jnp.array(self.energies)

    X = self.source.all_candidates
    Y = self.source.energies

    print('Y.shape', Y.shape)

    # Generate samples from the posterior on hyperparameters.
    hyper_rng, self.rng = jrnd.split(self.rng)
    t0 = time.time()
    ls_samples, amp_samples, chol_K_samples = slice_sample_hypers(
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
    print(jnp.mean(ls_samples, axis=0))
    print(jnp.mean(amp_samples, axis=0))
    
    # # Use the last values for the initialization next time.
    # self.init_ls = ls_samples[-1]
    # self.init_amp = amp_samples[-1]

    # # Chop off the burn-in samples.
    # ls_samples = ls_samples[self.cfg.chase.hypers.num_burnin:]
    # amp_samples = amp_samples[self.cfg.chase.hypers.num_burnin:]
    # chol_K_samples = chol_K_samples[self.cfg.chase.hypers.num_burnin:]

    # def apply_kernel(X1, X2, ls, amp):
    #   return amp * self.kernel_fn(X1/ls, X2/ls)

    # # Compute the posterior distribution on the energies.
    # def posterior_mean_cov(X, y, ls, amp, chol_K):
    #   k_Xx = apply_kernel(X, self.source.all_candidates, ls, amp)
    #   k_xx = apply_kernel(
    #       self.source.all_candidates,
    #       self.source.all_candidates,
    #       ls,
    #       amp,
    #     ) + jnp.eye(self.source.all_candidates.shape[0]) * self.cfg.gp.jitter
    #   solved = jax.scipy.linalg.cho_solve((chol_K, False), k_Xx)
    #   mean = jnp.dot(solved.T, y)
    #   cov = k_xx - jnp.dot(solved.T, k_Xx)
    #   chol_cov = jax.scipy.linalg.cholesky(cov)
    #   return mean, chol_cov

    # # Compute the posterior mean and covariance for each phase and each sample.
    # means, chol_covs = jax.vmap(
    #     jax.vmap(
    #         posterior_mean_cov,
    #         in_axes=(None, 1, 0, 0, 0),
    #       ),
    #     in_axes=(None, None, 0, 0, 0),
    #   )(X, Y, ls_samples, amp_samples, chol_K_samples)
    # t2 = time.time()
    # log.info("Posterior mean and covariance computation took %f seconds." % (t2-t1))

    # # Draw samples from all those posteriors.
    # def draw_posterior_samples(mean, chol_cov, rng, num_samples):
    #   Z = jrnd.normal(rng, (num_samples, mean.shape[0]))
    #   return mean + jnp.dot(chol_cov.T, Z.T).T
    
    # # Draw samples from the posterior for each phase and each sample.
    # posterior_samples = jax.vmap(
    #     jax.vmap(
    #         draw_posterior_samples,
    #         in_axes=(0, 0, 0, None),
    #         out_axes=1,
    #       ),
    #     in_axes=(0, 0, 0, None),
    #   )(
    #     means,
    #     chol_covs,
    #     jrnd.split(self.rng, amp_samples.shape),
    #     self.cfg.chase.posterior.num_samples,
    #   )
    # t3 = time.time()
    # log.info("Posterior sampling took %f seconds." % (t3-t2))

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
