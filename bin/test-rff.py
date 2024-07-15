import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import matplotlib.pyplot as plt
import time

#jax.config.update("jax_enable_x64", True)

from omegaconf import DictConfig

from apm.mcmc import slice_sample_hypers
from apm.gp.kernels import Matern52
from apm.utils import multivariate_t_rvs

log = logging.getLogger()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

  seed = 2
  rng = jrnd.PRNGKey(seed)
  num_data = 10
  
  ##############################################################################
  # Generate fake 1d data.
  data_x_rng, data_y_rng, rng = jrnd.split(rng, 3)
  data_x = jrnd.uniform(data_x_rng, (num_data, 1), minval=0.0, maxval=5.0)
  data_y = jnp.sin(data_x) + jrnd.normal(data_y_rng, (num_data, 1)) * 0.01
  grid_x = jnp.linspace(0.0, 5.0, 500)[:,jnp.newaxis]

  ##############################################################################
  # Initialize the hyperparameters.
  init_ls_rng, init_amp_rng, rng = jrnd.split(rng, 3)
  init_ls = jrnd.uniform(init_ls_rng, (1,1,),
      minval=cfg.gp.lengthscale_prior[0],
      maxval=cfg.gp.lengthscale_prior[1],
  )
  log.info("Initial lengthscales: %s" % init_ls)
  init_amp = jrnd.uniform(init_amp_rng, (1,),
      minval=cfg.gp.amplitude_prior[0],
      maxval=cfg.gp.amplitude_prior[1],
  )

  ##############################################################################
  # Sample from the posterior.
  hyper_rng, rng = jrnd.split(rng)
  t0 = time.time()
  ls_samples, amp_samples = slice_sample_hypers(
    hyper_rng,
    data_x,
    data_y,
    cfg,
    init_ls,
    init_amp,
    cfg.chase.hypers.num_samples + cfg.chase.hypers.num_burnin,
    cfg.chase.hypers.num_thin,  
  )
  t1 = time.time()
  log.info("Hyperparameter sampling took %f seconds." % (t1-t0))

  # Remove burn-in.
  ls_samples = ls_samples[cfg.chase.hypers.num_burnin:]
  amp_samples = amp_samples[cfg.chase.hypers.num_burnin:]

  ##############################################################################
  # For each of the hyperparameter samples, compute the posterior mean and var.
  kernel_fn = globals()[cfg.gp.kernel]
  def apply_kernel(X1, X2, ls, amp):
    return amp**2 * kernel_fn(X1/ls, X2/ls)
  
  def posterior_mean_cov(X, y, ls, amp):
    k_XX = apply_kernel(X, X, ls, amp)
    k_Xx = apply_kernel(X, grid_x, ls, amp)
    k_xx = apply_kernel(
      grid_x,
      grid_x,
      ls,
      amp,
    ) + jnp.eye(grid_x.shape[0]) * cfg.gp.jitter
    chol_K = jax.scipy.linalg.cho_factor(k_XX)
    solved = jax.scipy.linalg.cho_solve(chol_K, k_Xx)
    mean = jnp.dot(solved.T, y)
    cov = k_xx - jnp.dot(solved.T, k_Xx)
    chol_cov = jax.scipy.linalg.cholesky(cov)
    return mean, chol_cov

  means, chol_covs = jax.vmap(
    posterior_mean_cov,
    in_axes=(None, None, 0, 0),
  )(data_x, data_y, ls_samples, amp_samples)
  t2 = time.time()
  log.info("Posterior mean and covariance computation took %f seconds." % (t2-t1))

  ##############################################################################
  # Draw samples from the posteriors.
  def draw_posterior_samples(mean, chol_cov, rng, num_samples):
    Z = jrnd.normal(rng, (num_samples, mean.shape[0]))
    return mean.T + jnp.dot(chol_cov.T, Z.T).T

  post_rng, rng = jrnd.split(rng)
  posterior_samples = jax.vmap(
    draw_posterior_samples,
    in_axes=(0, 0, 0, None),
    out_axes=1,
  )(
    means,
    chol_covs,
    jrnd.split(rng, cfg.chase.hypers.num_samples),
    cfg.chase.posterior.num_samples,
  )
  t3 = time.time()
  log.info("Posterior sampling took %f seconds." % (t3-t2))
  samples = posterior_samples.reshape(-1, grid_x.shape[0])

  ##############################################################################
  # Draw basis functions for each hyperparameter.  We could use the same
  # underlying parameters for each hyperparameter, but it seems more robust
  # to use different ones.
  proj_rng, phase_rng, rng = jrnd.split(rng, 3)
  rff_projections = multivariate_t_rvs(
    proj_rng, 
    mu=jnp.zeros((1,)), 
    sigma=jnp.eye(1),
    df=5, # Matern 5/2
    shape=(cfg.chase.hypers.num_samples, cfg.gp.num_rff),
  ) 
  rff_phases = jrnd.uniform(
    phase_rng, 
    (cfg.chase.hypers.num_samples, cfg.gp.num_rff),
    minval=0.0,
    maxval=2*jnp.pi,
  )
  t4 = time.time()
  log.info("RFF sampling took %f seconds." % (t4-t3))

  ##############################################################################
  # Compute the basis functions on the grid for each hyperparameter.
  def rff_basis_function(x, projections, phases, ls, amp):
    rff = jnp.sqrt(2.0 / cfg.gp.num_rff) * jnp.cos(
      jnp.dot(projections, x/ls) + phases[:,jnp.newaxis]
    )
    return amp * rff.squeeze()

  grid_basis = \
    jax.vmap( # vmap over the grid
      jax.vmap( # vmap over hyperparameters
        rff_basis_function,
        in_axes=(None, 0, 0, 0, 0,)
      ),
      in_axes=(0, None, None, None, None),
    )(grid_x, rff_projections, rff_phases, ls_samples, amp_samples)
  
  data_basis = \
    jax.vmap( # vmap over the data
      jax.vmap( # vmap over hyperparameters
        rff_basis_function,
        in_axes=(None, 0, 0, 0, 0,)
      ),
      in_axes=(0, None, None, None, None),
    )(data_x, rff_projections, rff_phases, ls_samples, amp_samples)

  ##############################################################################
  # Compute the posteriors over the weights.
  def weight_posterior(basis, y, noise):
    iSigma = jnp.eye(cfg.gp.num_rff) + jnp.dot(basis.T, (1/noise) * basis)
    chol_iSigma = jax.scipy.linalg.cholesky(iSigma)
    mu = jnp.dot(jax.scipy.linalg.cho_solve((chol_iSigma, False), basis.T), y/noise)
    return mu, jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(cfg.gp.num_rff))

  wt_post_means, wt_post_chols = jax.vmap(
    weight_posterior,
    in_axes=(1, None, None),
  )(data_basis, data_y, cfg.gp.jitter)

  ##############################################################################
  # Sample from the posterior on weights.
  post_weight_rng, rng = jrnd.split(rng)
  def draw_weight_samples(mu, chol, rng, num_samples):
    Z = jrnd.normal(rng, (num_samples, mu.shape[0]))
    return mu + jnp.dot(chol.T, Z.T)
  
  weight_samples = jax.vmap(
    draw_weight_samples,
    in_axes=(0, 0, 0, None),
    out_axes=1,
  )(wt_post_means, wt_post_chols, jrnd.split(rng, cfg.chase.hypers.num_samples), cfg.chase.posterior.num_samples)
  #print(weight_samples.shape, grid_basis.shape)

  ##############################################################################
  # Evaluate the posterior samples on the grid.
  print(func_samples.shape)
  func_samples = jnp.einsum('ijk,lji->ljk', weight_samples, grid_basis)

  ##############################################################################
  # Estimate entropy
  import jax.scipy.stats as stats
  def entropy(values):
    kernel = stats.gaussian_kde(values)
    return -jnp.mean(kernel.logpdf(values))
  entropies = jax.vmap(entropy)(func_samples.reshape(func_samples.shape[0],-1))
  #print(entropies.shape)


  plt.figure(figsize=(5,10))
  plt.subplot(3, 1, 1)
  plt.plot(data_x, data_y, "o")
  plt.plot(grid_x, samples.T, "k", alpha=0.1)
  plt.ylim(-2, 2)

  plt.subplot(3, 1, 2)
  plt.plot(data_x, data_y, "o")
  plt.plot(grid_x, func_samples.reshape(grid_x.shape[0],-1), "k", alpha=0.1)
  plt.ylim(-2, 2)

  plt.subplot(3, 1, 3)
  plt.plot(grid_x, entropies)

  plt.savefig("test-rff.pdf")


if __name__ == "__main__":
  main()

