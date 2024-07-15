from typing import Callable, Any, Tuple
from jax.typing import ArrayLike
from jax import Array
from .types import PRNGKey

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc
import logging
import time

# FIXME: predict -> candidate

#from .gp.phase import generate_posterior_sampler, init_phase_gp
from .gp.kernels import Matern52
#from .hull import lower_hull_points
#from .utils import multivariate_t_rvs, entropy

from .gp.mcmc import generate_slice_sampler

log = logging.getLogger()

@jdc.pytree_dataclass
class CHASEState:
  lengthscales:    Array # num_phases x num_species [float32]
  amplitudes:      Array # num_phases [float32]
  data_basis:      Array # num_phases x num_data x num_rff [float32]
  cand_basis:      Array # num_phases x num_candidates x num_rff [float32]
  post_means:      Array # num_phases x num_rff [float32]
  post_chol_covs:  Array # num_phases x num_rff x num_rff [float32]
  post_wt_samples: Array # num_phases x num_rff x num_samples [float32]

################################################################################
# Temporarily putting things here until it's clear how to modularize.
from .utils import multivariate_t_rvs
  
def rff_basis_function(
    x:               ArrayLike, # num_species [float32]
    rff_projections: ArrayLike, # num_rff x num_species [float32]
    rff_phases:      ArrayLike, # num_rff [float32]
    lengthscales:    ArrayLike, # num_species [float32]
    amplitude:       ArrayLike, # [float32]
  ) -> Array: # num_rff [float32]

  num_rff = rff_projections.shape[0]
  rff = jnp.sqrt(2.0 / num_rff) * jnp.cos(
      jnp.dot(rff_projections, x/lengthscales) + rff_phases
    )
  assert rff.shape == (num_rff,)
  return amplitude * rff.squeeze()

@jax.jit
def weight_posterior_cholesky(
    basis_X:     ArrayLike, # num_data x num_rff [float32]
    y:           ArrayLike, # num_data [float32]
    noise:       ArrayLike, # num_data [boolean]
    prior_mean:  ArrayLike, # num_rff [float32]
    prior_chol:  ArrayLike, # num_rff x num_rff [float32]
  ) -> Tuple[Array, Array]: # num_rff [float32], num_rff x num_rff [float32]
  # TODO: explore an SVD alternative

  assert basis_X.shape[1] == prior_mean.shape[0]
  assert basis_X.shape[1] == prior_chol.shape[0]
  assert basis_X.shape[1] == prior_chol.shape[1]
  assert basis_X.shape[0] == y.shape[0]
  assert y.shape == noise.shape

  num_data = basis_X.shape[0]
  num_rff  = basis_X.shape[1]

  # Find the inverse of the prior using its Cholesky decomposition.
  prior_iSigma = jax.scipy.linalg.cho_solve((prior_chol, False), jnp.eye(num_rff))
  assert prior_iSigma.shape == (num_rff, num_rff)

  # Compute the inverse of the posterior covariance.
  iSigma = prior_iSigma + jnp.dot(basis_X.T, (1/noise[:,jnp.newaxis]) * basis_X)
  assert iSigma.shape == (num_rff, num_rff)

  # Take the Cholesky of the inverse to get the posterior covariance.
  chol_iSigma = jax.scipy.linalg.cholesky(iSigma)

  # FIXME there must be a better way to do this.
  # For some reason solve_triangular with identity on chol_iSigma doesn't
  # do the right thing.
  #chol_Sigma = jax.scipy.linalg.solve_triangular(chol_iSigma, jnp.eye(cfg.gp.num_rff))
  Sigma = jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(num_rff))
  chol_Sigma = jax.scipy.linalg.cholesky(Sigma)

  to_solve = jnp.dot(basis_X.T, y/noise) + jnp.dot(prior_iSigma, prior_mean)
  assert to_solve.shape == (num_rff,)

  #mu = Sigma @ to_solve
  mu = jax.scipy.linalg.cho_solve((chol_iSigma, False), to_solve)
  assert mu.shape == (num_rff,)

  return mu, chol_Sigma

def posterior_predictive_samples(
    rng:          PRNGKey,
    data_X:       ArrayLike,  # num_data x num_species [float32]
    data_Y:       ArrayLike,  # num_phases x num_data  [float32]
    data_noise:   ArrayLike,  # num_phases x num_data  [boolean]
    predict_X:    ArrayLike,  # num_predict x num_species [float32]
    lengthscales: ArrayLike,  # num_phases x num_species [float32]
    amplitudes:   ArrayLike,  # num_phases [float32]
    num_rff:      int,
    num_samples:  int,
  ) -> CHASEState:

  assert data_X.shape[1] == predict_X.shape[1]
  assert data_X.shape[1] == lengthscales.shape[1]
  assert data_Y.shape[1] == data_X.shape[0]
  assert data_Y.shape[0] == lengthscales.shape[0]
  assert data_Y.shape[0] == amplitudes.shape[0]
  assert data_Y.shape == data_noise.shape

  num_data    = data_X.shape[0]
  num_phases  = data_Y.shape[0]
  num_species = data_X.shape[1]
  num_predict = predict_X.shape[0]

  proj_rng, phase_rng, sample_rng, rng = jrnd.split(rng, 4)

  # num_phases x num_rff x num_phases
  rff_projections = multivariate_t_rvs(
      proj_rng, 
      mu=jnp.zeros(num_species),
      sigma=jnp.eye(num_species),
      df=5, # Matern 5/2 <--- generalize
      shape=(num_phases, num_rff),
    )
  assert rff_projections.shape == (num_phases, num_rff, num_species)
  
  # num_phases x num_rff
  rff_phases = jrnd.uniform(
      phase_rng, 
      (num_phases, num_rff,),
      minval=0.0,
      maxval=2*jnp.pi,
    )
  assert rff_phases.shape == (num_phases, num_rff)

  # num_phases x num_data x num_rff
  data_basis = \
    jax.vmap( # vmap over phases
      jax.vmap( # vmap over the data
        rff_basis_function,
        in_axes=(0, None, None, None, None),
      ),
      in_axes=(None, 0, 0, 0, 0,),
    )(
      data_X,          # num_data x num_species
      rff_projections, # num_phases x num_rff x num_species
      rff_phases,      # num_phases x num_rff
      lengthscales,    # num_phases x num_species
      amplitudes,      # num_phases
      )
  assert data_basis.shape == (num_phases, num_data, num_rff)

  # num_phases x num_predict x num_rff
  predict_basis = \
    jax.vmap( # vmap over phases
      jax.vmap( # vmap over the prediction points
        rff_basis_function,
        in_axes=(0, None, None, None, None),
      ),
      in_axes=(None, 0, 0, 0, 0,),
    )(
      predict_X,       # num_predict x num_species
      rff_projections, # num_phases x num_rff x num_species
      rff_phases,      # num_phases x num_rff
      lengthscales,    # num_phases x num_species
      amplitudes,      # num_phases
      )
  assert predict_basis.shape == (num_phases, num_predict, num_rff)

  # Basic prior for the RFF weights.
  prior_mean = jnp.zeros(num_rff)
  prior_chol = jnp.eye(num_rff)

  # Compute the posterior for each phase.
  post_means, post_chol_covs = jax.vmap( # over phases
    weight_posterior_cholesky,
    in_axes=(0, 0, 0, None, None,),
  )(
    data_basis, # num_phases x num_data x num_rff
    data_Y,     # num_phases x num_data
    data_noise, # num_phases x num_data
    prior_mean, # num_rff
    prior_chol, # num_rff x num_rff
  )
  assert post_means.shape == (num_phases, num_rff)
  assert post_chol_covs.shape == (num_phases, num_rff, num_rff)

  # Generate samples from the posterior on weights.
  post_wt_samples = jax.vmap( # over phases
    lambda z, mu, chol: mu[:,jnp.newaxis] + jnp.dot(chol.T, z),
    in_axes=(0, 0, 0,),
  )(
    jrnd.normal(sample_rng, (num_phases, num_rff, num_samples)),
    post_means,     # num_phases x num_rff
    post_chol_covs, # num_phases x num_rff x num_rff
  )
  assert post_wt_samples.shape == (num_phases, num_rff, num_samples)

  # Construct a state object.
  return CHASEState(
      lengthscales=lengthscales,
      amplitudes=amplitudes,
      data_basis=data_basis,
      cand_basis=predict_basis,
      post_means=post_means,
      post_chol_covs=post_chol_covs,
      post_wt_samples=post_wt_samples,
    )

from .gp.ess import generate_ess_sampler

def hull_conditioned_samples(
    rng:           PRNGKey,
    candidates:    ArrayLike,  # num_candidates x num_species [float32]
    state:         CHASEState,
    jitter:        float,
    noise_missing: float,
  ) -> Array: 

  num_candidates = candidates.shape[0]
  num_phases     = state.post_wt_samples.shape[0]
  num_samples    = state.post_wt_samples.shape[2]
  num_rff        = state.post_means.shape[1]

  ess_sampler = generate_ess_sampler(
    candidates,       # num_candidates x num_species
    state.cand_basis, # num_phases x num_candidates x num_rff
    )

  # VMAP this over samples
  func_samples = jax.vmap(
    ess_sampler,
    in_axes=(0, 2, None, None, None, None, None,),
    out_axes=2,
    )(
      jrnd.split(rng, num_samples),
      state.post_wt_samples,
      state.post_means,
      state.post_chol_covs,
      13,
      jitter,
      noise_missing,
    )

  return func_samples

################################################################################

class CHASE:
  def __init__(self, cfg, source, rng):

    # Global Hydra configuration.
    self.cfg = cfg

    # Data source.
    self.source = source

    # Random number generator.
    self.rng = rng

    # Indices of compositions evaluated so far.
    self.comps = []

    # Energies of compositions evaluated so far.
    # num_data x num_phases
    self.energies = []

    # Boolean mask indicating which phases have been evaluated.
    # This is because not all phases will be evaluated for every composition.
    # num_data x num_phases
    self.evaluated = []

    self._init_hypers()

    # The generator pattern makes it easier to vmap this.
    self.slice_sampler = generate_slice_sampler(
        globals()[cfg.gp.kernel],  # <---- FIXME, this is ugly.
        cfg.gp.lengthscale_prior,
        cfg.gp.amplitude_prior,
        cfg.chase.hypers.num_samples + cfg.chase.hypers.num_burnin,
        cfg.chase.hypers.num_thin,
      )     

  ##############################################################################
  def _init_hypers(self):
    ''' Initialize the hyperparameters with a draw from the prior. '''

    init_ls_rng, init_amp_rng, self.rng = jrnd.split(self.rng, 3)

    self.init_ls = jrnd.uniform(
        init_ls_rng,
        (self.source.num_phases, self.source.num_species,),
        minval=self.cfg.gp.lengthscale_prior[0],
        maxval=self.cfg.gp.lengthscale_prior[1],
      )
    log.info("Initial lengthscales:\n%s" % self.init_ls)

    self.init_amp = jrnd.uniform(
        init_amp_rng, (self.source.num_phases,),
        minval=self.cfg.gp.amplitude_prior[0],
        maxval=self.cfg.gp.amplitude_prior[1],
      )
    log.info("Initial amplitudes:\n%s" % self.init_amp)

  ##############################################################################
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
      phase_evaluated = []
      for jj in range(self.source.num_phases):
        phase_energies.append(self.source.eval_phase(idx,jj))
        phase_evaluated.append(True)
      self.energies.append(jnp.array(phase_energies).ravel())
      self.evaluated.append(jnp.array(phase_evaluated).ravel())
      log.info("Pure composition %d has energies %s." % (ii+1, self.energies[-1]))

  ##############################################################################  
  def select_next(self):
    ''' Select the next composition to evaluate. '''

    data_X = self.source.all_candidates[jnp.array(self.comps).ravel(),:]
    data_Y = jnp.array(self.energies).T

    # Consider non-evaluated phases to simply have large uncertainty.
    # This is effectively padding things to allow for vmapping.
    data_noise = jnp.where(
        jnp.array(self.evaluated).T,
        self.cfg.gp.jitter,
        self.cfg.gp.noise_missing,
      )

    # Draw samples from the hyperparameter posterior.
    ls_samples, amp_samples = self.hyper_sample(data_X, data_Y, data_noise)
    assert ls_samples.shape == (self.cfg.chase.hypers.num_samples, self.source.num_phases, self.source.num_species)
    assert amp_samples.shape == (self.cfg.chase.hypers.num_samples, self.source.num_phases)

    # Just loop for now and aggregate samples.
    posterior_funcs = []
    post_hull_funcs = []
    for ii in range(ls_samples.shape[0]):
      predictive_rng, hull_rng, self.rng = jrnd.split(self.rng, 3)

      posterior_state = posterior_predictive_samples(
          predictive_rng,
          data_X,
          data_Y,
          data_noise,
          self.source.all_candidates,
          ls_samples[ii,:,:],
          amp_samples[ii,:],
          self.cfg.gp.num_rff,
          self.cfg.chase.posterior.num_samples,
        )

      hull_samples = hull_conditioned_samples(
          hull_rng,
          self.source.all_candidates,
          posterior_state,
          self.cfg.gp.jitter,
          self.cfg.gp.noise_missing
        )
      post_hull_funcs.append(hull_samples)

      # Compute posterior functions.
      post_funcs = jnp.einsum(
          'ijk,ilj->ilk',
          posterior_state.post_wt_samples, # num_phases x num_rff x num_samples
          posterior_state.cand_basis, # num_phases x num_candidates x num_rff
        )
      posterior_funcs.append(post_funcs)

      import matplotlib.pyplot as plt
      plt.figure(figsize=(10,10))

      colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
      for jj in range(self.source.num_phases):
        plt.plot(self.source.all_candidates[:,0], post_funcs[jj,:,0], color=colors[jj], alpha=0.5)
        plt.plot(data_X[:,0], data_Y[jj,:], 'o', color=colors[jj])

        plt.plot(self.source.all_candidates[:,0], hull_samples[jj,:,0,:], color=colors[jj], alpha=0.2)

      plt.savefig("mock-%d.png" % (ii))

    post_hull_funcs = jnp.stack(post_hull_funcs, axis=-1)
    post_funcs = jnp.stack(posterior_funcs, axis=-1)
    print('post_hull_funcs', post_hull_funcs.shape)
    print('post_funcs', post_funcs.shape)

    from .utils import entropy

    # Compute entropy of the posterior samples without hull conditioning.
    post_funcs = post_funcs.reshape(*post_funcs.shape[:2], -1)
    post_entropies = jax.vmap( # phases
      jax.vmap( # candidates
        entropy,
        in_axes=(0,),
      ),
      in_axes=(0,),
    )(post_funcs)
    print('post_entropies', post_entropies.shape)

    # Compute entropy per hull-conditioned sample.
    post_hull_funcs = post_hull_funcs.transpose(0, 1, 2, 4, 3)
    post_hull_funcs = post_hull_funcs.reshape(*post_hull_funcs.shape[:2], -1, post_hull_funcs.shape[-1])
    post_hull_entropies = jax.vmap( # phases
      jax.vmap( # candidates
        jax.vmap( # post + hyper samples
          entropy,
          in_axes=(0,),
        ),
        in_axes=(0,),
      ),
      in_axes=(0,),
    )(post_hull_funcs)
    print('post_hull_entropies', post_hull_entropies.shape)
    
    # Average over post+hyper samples.
    post_hull_entropies = jnp.mean(post_hull_entropies, axis=-1)

    plt.figure(figsize=(10,10))
    for jj in range(self.source.num_phases):
      plt.subplot(self.source.num_phases+1, 1, jj+1)
      plt.plot(self.source.all_candidates[:,0], post_entropies[jj,:], color='green')
      plt.plot(self.source.all_candidates[:,0], post_hull_entropies[jj,:], color='cyan')
      plt.title("Phase %d" % (jj+1))

    diffs = post_entropies - post_hull_entropies
    plt.subplot(self.source.num_phases+1, 1, self.source.num_phases+1)
    plt.plot(self.source.all_candidates[:,0], diffs[0,:], color='red')
    plt.plot(self.source.all_candidates[:,0], diffs[1,:], color='blue')

    plt.savefig("mock-entropies.png")







  ##############################################################################
  def hyper_sample(self, X, Y, noise):
    ''' Sample hyperparameters from the posterior. '''

    hyper_rng, self.rng = jrnd.split(self.rng)

    # vmap over phases
    ls_samples, amp_samples = jax.vmap(
      self.slice_sampler,
      in_axes=(0, None, 0, 0, 0, 0,),
      )(
        jrnd.split(hyper_rng, self.source.num_phases),
        X, Y, noise,
        self.init_ls, self.init_amp,
      )
    
    # Remove burnin and reshape so that the first axis is the sample axis.
    ls_samples = ls_samples[:,self.cfg.chase.hypers.num_burnin:,:].transpose(1,0,2)
    amp_samples = amp_samples[:,self.cfg.chase.hypers.num_burnin:].transpose(1,0)

    # Store last samples as the new initial values.
    self.init_ls = ls_samples[:,-1,:]
    self.init_amp = amp_samples[:,-1]

    return ls_samples, amp_samples

    




