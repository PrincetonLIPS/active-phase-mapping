# Computations performed per phase.
import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import jax.random as jrnd

from .mcmc import generate_slice_sampler
from ..utils import multivariate_t_rvs

@jdc.pytree_dataclass
class PhaseGP:
  lengthscales: jnp.ndarray
  amplitude: jnp.ndarray

def init_phase_gp(rng, num_phases, num_species, ls_prior, amp_prior):
  ls_rng, amp_rng, rng = jax.random.split(rng, 3)
  lengthscales = jax.random.uniform(
      rng, 
      (num_phases, num_species,),
      minval=ls_prior[0], 
      maxval=ls_prior[1],
    )
  amplitude = jax.random.uniform(
      rng, 
      (num_phases,),
      minval=amp_prior[0],
      maxval=amp_prior[1],
    )

  return PhaseGP(lengthscales, amplitude)

def generate_posterior_sampler(cfg):

  slice_sampler = generate_slice_sampler(cfg)

  def posterior_sampler(rng, train_X, train_y, predict_x, state):

    # Zero-center the data.
    mean_y = jnp.mean(train_y)
    train_y = train_y - mean_y

    ############################################################################
    # Hyperparameter inference
    slice_rng, rng = jax.random.split(rng)
    ls_samples, amp_samples = slice_sampler(
        slice_rng,
        train_X,
        train_y,
        state.lengthscales[0,:], # FIXME
        state.amplitude[0],
        cfg.chase.hypers.num_samples + cfg.chase.hypers.num_burnin,
        cfg.chase.hypers.num_thin,
    )

    # Use the last values for the initialization next time.
    init_ls  = ls_samples[-1]
    init_amp = amp_samples[-1]

    # Chop off the burn-in samples.
    ls_samples = ls_samples[cfg.chase.hypers.num_burnin:]
    amp_samples = amp_samples[cfg.chase.hypers.num_burnin:]

    ############################################################################
    # Random Fourier features
    proj_rng, phase_rng, rng = jrnd.split(rng, 3)
    rff_projections = multivariate_t_rvs(
        proj_rng, 
        mu=jnp.zeros((train_X.shape[1],)), 
        sigma=jnp.eye(train_X.shape[1]),
        df=5, # Matern 5/2
        shape=(ls_samples.shape[0], cfg.gp.num_rff,),
      )   
    rff_phases = jrnd.uniform(
        phase_rng, 
        (ls_samples.shape[0], cfg.gp.num_rff,),
        minval=0.0,
        maxval=2*jnp.pi,
      )
    
    def rff_basis_function(x, rff_projections, rff_phases, ls, amp):
      rff = jnp.sqrt(2.0 / cfg.gp.num_rff) * jnp.cos(
          jnp.dot(rff_projections, x/ls) + rff_phases
        )
      return amp * rff.squeeze()

    predict_basis = \
      jax.vmap( # vmap over the grid
        jax.vmap( # vmap over hyperparameters
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(predict_x, rff_projections, rff_phases, ls_samples, amp_samples)
    
    train_basis = \
      jax.vmap( # vmap over the data
        jax.vmap( # vmap over hyperparameters
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(train_X, rff_projections, rff_phases, ls_samples, amp_samples)

    ############################################################################
    # Compute the posterior on weights
    def weight_posterior(basis, y, noise):
      iSigma = jnp.eye(cfg.gp.num_rff) + jnp.dot(basis.T, (1/noise) * basis)
      chol_iSigma = jax.scipy.linalg.cholesky(iSigma)
      mu = jnp.dot(jax.scipy.linalg.cho_solve((chol_iSigma, False), basis.T), y/noise)
      return mu, jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(cfg.gp.num_rff))

    wt_post_means, wt_post_chols = jax.vmap(
      weight_posterior,
      in_axes=(1, None, None),
    )(train_basis, train_y, cfg.gp.jitter)

    ############################################################################
    # Sample from the posterior on weights
    post_weight_rng, rng = jrnd.split(rng)
    def draw_weight_samples(mu, chol, rng, num_samples):
      Z = jrnd.normal(rng, (mu.shape[0], num_samples))
      return (mu[:,jnp.newaxis] + jnp.dot(chol.T, Z)).T
    
    weight_samples = jax.vmap(
      draw_weight_samples,
      in_axes=(0, 0, 0, None),
      out_axes=1,
    )(wt_post_means, wt_post_chols, 
      jrnd.split(rng, cfg.chase.hypers.num_samples),
      cfg.chase.posterior.num_samples,
      )

    ############################################################################
    # Weigh the basis functions with the samples.
    func_samples = jnp.einsum('ijk,ljk->ijl', weight_samples, predict_basis)
    func_samples = func_samples.reshape(-1,predict_x.shape[0]) + mean_y

    return func_samples

  return posterior_sampler

  
