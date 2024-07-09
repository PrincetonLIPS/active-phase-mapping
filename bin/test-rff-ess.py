import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import matplotlib.pyplot as plt
import time

#jax.config.update("jax_enable_x64", True)

from omegaconf import DictConfig

from apm.gp.mcmc import generate_slice_sampler
from apm.gp.kernels import Matern52
from apm.utils import multivariate_t_rvs
from apm.hull import lower_hull_points

log = logging.getLogger()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

  seed = 2
  rng = jrnd.PRNGKey(seed)
  num_data = 3
  num_phases = 2
  
  ##############################################################################
  # Generate fake 1d data.
  data_x_rng, data_y_rng, rng = jrnd.split(rng, 3)
  data_x = jrnd.uniform(data_x_rng, (num_data, 1), minval=0.0, maxval=5.0)
  data_y = jrnd.normal(data_y_rng, (num_data, num_phases))
  grid_x = jnp.linspace(0.0, 5.0, 500)[:,jnp.newaxis]

  ##############################################################################
  # Initialize the hyperparameters.
  init_ls_rng, init_amp_rng, rng = jrnd.split(rng, 3)
  init_ls = jrnd.uniform(init_ls_rng, (num_phases,1,),
      minval=cfg.gp.lengthscale_prior[0],
      maxval=cfg.gp.lengthscale_prior[1],
  )
  log.info("Initial lengthscales: %s" % init_ls)
  init_amp = jrnd.uniform(init_amp_rng, (num_phases,),
      minval=cfg.gp.amplitude_prior[0],
      maxval=cfg.gp.amplitude_prior[1],
  )
  log.info("Initial amplitudes: %s" % init_amp)

  ##############################################################################
  # Sample from the posterior on hyperparameters, for all phases.
  slice_sampler = generate_slice_sampler(cfg)
  hyper_rng, rng = jrnd.split(rng)
  ls_samples, amp_samples = jax.vmap(slice_sampler, in_axes=(0,None,1,0,0,None,None))(
    jrnd.split(hyper_rng, num_phases),
    data_x,
    data_y,
    init_ls,
    init_amp,
    cfg.chase.hypers.num_samples + cfg.chase.hypers.num_burnin,
    cfg.chase.hypers.num_thin,  
  )

  # Eliminate burn-in samples and reshape.
  ls_samples = ls_samples[:,cfg.chase.hypers.num_burnin:,:].transpose(1,0,2)
  amp_samples = amp_samples[:,cfg.chase.hypers.num_burnin:].transpose(1,0)
  # print(ls_samples.shape, amp_samples.shape)

  ##############################################################################
  # Iterate over each hyperparameter sample rather than vmapping to save memory.
  for ii in range(ls_samples.shape[0]):
    ls = ls_samples[ii]
    amp = amp_samples[ii]

    ############################################################################
    # Random Fourier features
    proj_rng, phase_rng, rng = jrnd.split(rng, 3)
    rff_projections = multivariate_t_rvs(
        proj_rng, 
        mu=jnp.zeros((data_x.shape[1],)), 
        sigma=jnp.eye(data_x.shape[1]),
        df=5, # Matern 5/2
        shape=(num_phases, cfg.gp.num_rff,),
      )   
    rff_phases = jrnd.uniform(
        phase_rng, 
        (num_phases, cfg.gp.num_rff,),
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
        jax.vmap( # vmap over phases
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(grid_x, rff_projections, rff_phases, ls, amp)

    train_basis = \
       jax.vmap( # vmap over the grid
        jax.vmap( # vmap over phases
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(data_x, rff_projections, rff_phases, ls, amp)
    print('bases', train_basis.shape, predict_basis.shape)
    
    ############################################################################
    # Compute the posterior on weights.
    def weight_posterior(basis, y, noise):
      iSigma = jnp.eye(cfg.gp.num_rff) + jnp.dot(basis.T, (1/noise) * basis)
      chol_iSigma = jax.scipy.linalg.cholesky(iSigma)
      mu = jnp.dot(jax.scipy.linalg.cho_solve((chol_iSigma, False), basis.T), y/noise)
      return mu, jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(cfg.gp.num_rff))

    wt_post_means, wt_post_chols = jax.vmap( # over phases
      weight_posterior,
      in_axes=(1, 1, None),
    )(train_basis, data_y, cfg.gp.jitter)
    # print(wt_post_means.shape, wt_post_chols.shape)

    ############################################################################
    # Sample from the posterior on weights.
    post_weight_rng, rng = jrnd.split(rng)
    def draw_weight_samples(mu, chol, rng, num_samples):
      Z = jrnd.normal(rng, (mu.shape[0], num_samples))
      return (mu[:,jnp.newaxis] + jnp.dot(chol.T, Z)).T
    
    weight_samples = jax.vmap( # vmap over phases
      draw_weight_samples,
      in_axes=(0, 0, 0, None),
    )(wt_post_means, wt_post_chols, jrnd.split(rng, (num_phases,)),
      cfg.chase.posterior.num_samples,
    )
    #print(weight_samples.shape)
    # We'll use these later to compute entropy.

    ############################################################################
    # Get the posterior functions themselves.
    func_samples = jnp.einsum('ijk,lik->jli', weight_samples, predict_basis)
    #print(func_samples.shape)

    ############################################################################
    # Compute the convex hull of the mins.
    def convex_hull(X, Y):
      return lower_hull_points(X, jnp.min(Y, axis=1))
    
    tight = jax.vmap(
      convex_hull,
      in_axes=(None,0),
    )(grid_x, func_samples)



    ############################################################################
    # Sample conditioned on the hull being the same.
    def hull_same_x(weights, target):
      funcs = jnp.einsum('ik,lik->li', weights, predict_basis)
      tight = convex_hull(grid_x, funcs)
      return jnp.all(tight == target)
    
    def ess_step(rng, cur_wts, mean, chol_cov, target):
      bound_rng, nu_rng, rng = jrnd.split(rng, 3)
      init_upper = jrnd.uniform(bound_rng, minval=0, maxval=2*jnp.pi)
      init_lower = init_upper - 2*jnp.pi

      def _while_cond(state):
        rng, upper, lower, cur, nu, theta = state
        return jnp.isnan(theta)
      
      def _while_body(state):
        rng, upper, lower, cur, nu, _ = state

        theta_rng, rng = jrnd.split(rng)

        theta = jrnd.uniform(theta_rng, minval=lower, maxval=upper)

        # Compute new weights.
        new_wts = jnp.cos(theta)*cur + jnp.sin(theta)*nu + mean

        # Check the hull.
        done = hull_same_x(new_wts, target)

        # Shrink the bracket.
        upper = jnp.where(theta > 0, theta, upper)
        lower = jnp.where(theta < 0, theta, lower)

        return (rng, upper, lower, cur, nu, jnp.where(done, theta, jnp.nan))
      
      # Need to vmap this one over phases.
      def draw_phase_nu(rng, chol):
        Z = jrnd.normal(rng, (chol.shape[0],))
        return jnp.dot(chol.T, Z)

      nu_rng, rng = jrnd.split(rng)
      nu = jax.vmap(
        draw_phase_nu,
        in_axes=(0, 0),
      )(jrnd.split(nu_rng, num_phases), chol_cov)

      _, _, _, _, _, theta = jax.lax.while_loop(
        _while_cond,
        _while_body,
        (rng, init_upper, init_lower, cur_wts-mean, nu, jnp.nan),
      )

      new_wts = jnp.cos(theta)*(cur_wts-mean) + jnp.sin(theta)*nu + mean

      return new_wts

    print('weight_samples', weight_samples.shape)

    idx = 1

    def _ess_scan(carry, _):
      rng, cur_wts = carry
      step_rng, rng = jrnd.split(rng)
      new_wts = ess_step(step_rng, cur_wts, wt_post_means, wt_post_chols, tight[idx])
      return (rng, new_wts), new_wts
    
    ess_rng, rng = jrnd.split(rng)
    _, hull_wts = jax.lax.scan(
      _ess_scan,
      (ess_rng, weight_samples[:,idx,:]),
      jnp.arange(100),
    )
    print('hull_wts', hull_wts.shape)

    # Look at a sample that has the same hull.
    funcs = jnp.einsum('ijk,ljk->ijl', hull_wts, predict_basis)
    print('funcs', funcs.shape)

    ############################################################################
    # Make a plot to look at.
    plt.figure(figsize=(10,10))
    plt.plot(grid_x, func_samples[idx,:,0].T, color='blue', alpha=0.5)
    plt.plot(grid_x, func_samples[idx,:,1].T, color='red', alpha=0.5)
    plt.plot(data_x, data_y, 'o')

    plt.plot(grid_x, funcs[:,0,:].T, color='blue', alpha=0.1)
    plt.plot(grid_x, funcs[:,1,:].T, color='red', alpha=0.1)

    plt.plot(grid_x[tight[idx]], 
             jnp.min(func_samples[idx,:,:], axis=1)[tight[idx]],
             'kx',
    )
    plt.savefig("rff-ess-%d.png" % ii)

if __name__ == "__main__":
  main()

