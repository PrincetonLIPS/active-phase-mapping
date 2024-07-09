import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)

from omegaconf import DictConfig

from apm.gp.mcmc import generate_slice_sampler
from apm.gp.kernels import Matern52
from apm.utils import multivariate_t_rvs
from apm.hull import lower_hull_points

log = logging.getLogger()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

  seed = 1
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
    def weight_posterior(basis, y, noise, prior_mean, prior_chol):
      noise = jnp.atleast_1d(noise)

      #prior_iSigma = jax.scipy.linalg.cho_solve((prior_chol, False), jnp.eye(cfg.gp.num_rff))


      # STUPID
      prior_iSigma = jnp.linalg.inv(prior_chol.T @ prior_chol)
      iSigma = prior_iSigma + jnp.dot(basis.T, (1/noise[:,jnp.newaxis]) * basis)
      Sigma = jnp.linalg.inv(iSigma)
      to_solve = jnp.dot(basis.T, y/noise) + jnp.dot(prior_iSigma, prior_mean)
      mu = Sigma @ to_solve

      #return mu, jax.scipy.linalg.inv(chol_iSigma) # FIXME
      return mu, jax.scipy.linalg.cholesky(Sigma)

    wt_post_means, wt_post_chols = jax.vmap( # over phases
      weight_posterior,
      in_axes=(1, 1, None, None, None),
    )(train_basis, data_y, cfg.gp.jitter, jnp.zeros((cfg.gp.num_rff,)), jnp.eye(cfg.gp.num_rff))
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
    )(wt_post_means, wt_post_chols, jrnd.split(post_weight_rng, (num_phases,)),
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
      min_Y = jnp.min(Y, axis=1)
      tight = lower_hull_points(X, min_Y)
      return tight[:,jnp.newaxis] * (min_Y[:,jnp.newaxis] == Y)
    
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

    # One approach: condition on which points are tight. In that case, we use
    # the posterior on weights to draw samples, and then effectively slice
    # off any part of the weight space that leads to a different set of points
    # being tight.  I think this is elegant, but it does not like to move very
    # much, even away from the tight points.
    #ess_means = wt_post_means
    #ess_chols = wt_post_chols

    # Another approach: condition on the hull being exactly the same.  This
    # means computing a posterior that accounts for the actual values of the
    # tight points.  This requires a new mean and covariance.
    def tight_params(tight, funcs, means, chols):

      # All phases together
      tight_noise = jnp.where(tight, cfg.gp.jitter, cfg.gp.noise_missing)

      print('tight', tight.shape, funcs.shape, tight_noise.shape, means.shape, chols.shape)
      # Treat the old params as the prior and these as "data".
      tight_means, tight_chols = jax.vmap( # over phases
          weight_posterior,
          in_axes=(1, 1, 1, 0, 0),
        )(predict_basis, funcs, tight_noise, means, chols)
      
      return tight_means, tight_chols

    print('predict_basis', predict_basis.shape)

    idx = 2

    # vmap over posterior samples
    tight_means, tight_chols = jax.vmap(
      tight_params,
      in_axes=(0, 0, None, None),
    )(tight, func_samples, wt_post_means, wt_post_chols)
    ess_means = tight_means[idx]
    ess_chols = tight_chols[idx]
    print('tight_means', tight_means.shape, tight_chols.shape)

    def _ess_scan(carry, _):
      rng, cur_wts = carry
      step_rng, rng = jrnd.split(rng)
      new_wts = ess_step(step_rng, cur_wts, ess_means, ess_chols, tight[idx])
      return (rng, new_wts), new_wts
    
    ess_rng, rng = jrnd.split(rng)
    _, hull_wts = jax.lax.scan(
      _ess_scan,
      (ess_rng, weight_samples[:,idx,:]),
      jnp.arange(50),
    )
    print('hull_wts', hull_wts.shape)

    # Look at a sample that has the same hull.
    funcs = jnp.einsum('ijk,ljk->ijl', hull_wts, predict_basis)
    print('funcs', funcs.shape)

    # sample directly
    def foo(rng, mean, chol):
      Z = jrnd.normal(rng, (chol.shape[0],))
      return jnp.dot(chol.T, Z) + mean
    foo_wts = jax.vmap(jax.vmap(
      foo,
      in_axes=(0, None, None),
    ), in_axes=(0, 0, 0))(jrnd.split(rng, (num_phases, 10)), ess_means, ess_chols)

    print('foo_wts', foo_wts.shape)
    print('predict_basis', predict_basis.shape)
    foo_funcs = jnp.einsum('ijk,lik->lij', foo_wts, predict_basis)

    meanfunc = jnp.einsum('ik,lik->li', ess_means, predict_basis)

    anytight = jnp.any(tight, axis=2)

    ############################################################################
    # Make a plot to look at.
    plt.figure(figsize=(10,10))
    plt.plot(grid_x, func_samples[idx,:,0].T, color='blue', alpha=0.5)
    plt.plot(grid_x, func_samples[idx,:,1].T, color='red', alpha=0.5)
    plt.plot(data_x, data_y, 'o')

    plt.plot(grid_x, funcs[:,0,:].T, color='blue', alpha=0.1)
    plt.plot(grid_x, funcs[:,1,:].T, color='red', alpha=0.1)

    #plt.plot(grid_x, meanfunc, '--')

    plt.plot(grid_x[anytight[idx]], 
             jnp.min(func_samples[idx,:,:], axis=1)[anytight[idx]],
             'kx',
    )
    plt.savefig("rff-ess-%d.png" % ii)

if __name__ == "__main__":
  main()

