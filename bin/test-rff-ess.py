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

  # num_data x num_species
  data_x = jrnd.uniform(data_x_rng, (num_data, 1), minval=0.0, maxval=5.0)

  # num_data x num_phases
  data_y = jrnd.normal(data_y_rng, (num_data, num_phases))

  # num_candidates x num_species
  grid_x = jnp.linspace(0.0, 5.0, 300)[:,jnp.newaxis]

  ##############################################################################
  # Initialize the hyperparameters.
  init_ls_rng, init_amp_rng, rng = jrnd.split(rng, 3)

  # num_phases x num_species
  init_ls = jrnd.uniform(init_ls_rng, (num_phases,1,),
      minval=cfg.gp.lengthscale_prior[0],
      maxval=cfg.gp.lengthscale_prior[1],
  )
  log.info("Initial lengthscales: %s" % init_ls)

  # num_phases x 1
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

  # num_samples x num_phases x num_species
  ls_samples = ls_samples[:,cfg.chase.hypers.num_burnin:,:].transpose(1,0,2)

  # num_samples x num_phases
  amp_samples = amp_samples[:,cfg.chase.hypers.num_burnin:].transpose(1,0)

  ##############################################################################
  # Iterate over each hyperparameter sample rather than vmapping to save memory.
  for ii in range(ls_samples.shape[0]):
    ls = ls_samples[ii]
    amp = amp_samples[ii]

    ############################################################################
    # Random Fourier features
    proj_rng, phase_rng, rng = jrnd.split(rng, 3)

    # num_phases x num_rff x num_species
    rff_projections = multivariate_t_rvs(
        proj_rng, 
        mu=jnp.zeros((data_x.shape[1],)), 
        sigma=jnp.eye(data_x.shape[1]),
        df=5, # Matern 5/2
        shape=(num_phases, cfg.gp.num_rff,),
      )   
    
    # num_phases x num_rff
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

    # num_candidates x num_phases x num_rff
    predict_basis = \
      jax.vmap( # vmap over the grid
        jax.vmap( # vmap over phases
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(grid_x, rff_projections, rff_phases, ls, amp)

    # num_data x num_phases x num_rff
    train_basis = \
       jax.vmap( # vmap over the grid
        jax.vmap( # vmap over phases
          rff_basis_function,
          in_axes=(None, 0, 0, 0, 0,)
        ),
        in_axes=(0, None, None, None, None),
      )(data_x, rff_projections, rff_phases, ls, amp)
    
    ############################################################################
    # Compute the posterior on weights.
    def weight_posterior(basis, y, noise, prior_mean, prior_chol):
      noise = jnp.atleast_1d(noise)

      prior_iSigma = jax.scipy.linalg.cho_solve((prior_chol, False), jnp.eye(cfg.gp.num_rff))
      iSigma = prior_iSigma + jnp.dot(basis.T, (1/noise[:,jnp.newaxis]) * basis)

      chol_iSigma = jax.scipy.linalg.cholesky(iSigma)

      # FIXME there must be a better way to do this.
      # For some reason solve_triangular with identity on chol_iSigma doesn't
      # do the right thing.
      #chol_Sigma = jax.scipy.linalg.solve_triangular(chol_iSigma, jnp.eye(cfg.gp.num_rff))
      Sigma = jax.scipy.linalg.cho_solve((chol_iSigma, False), jnp.eye(cfg.gp.num_rff))
      chol_Sigma = jax.scipy.linalg.cholesky(Sigma)

      to_solve = jnp.dot(basis.T, y/noise) + jnp.dot(prior_iSigma, prior_mean)

      #mu = Sigma @ to_solve
      mu = jax.scipy.linalg.cho_solve((chol_iSigma, False), to_solve)

      #return mu, jax.scipy.linalg.inv(chol_iSigma) # FIXME
      return mu, chol_Sigma

    # num_phases x num_rff
    # num_phases x num_rff x num_rff
    wt_post_means, wt_post_chols = jax.vmap( # over phases
      weight_posterior,
      in_axes=(1, 1, None, None, None),
    )(train_basis, data_y, cfg.gp.jitter, jnp.zeros((cfg.gp.num_rff,)), jnp.eye(cfg.gp.num_rff))

    ############################################################################
    # Sample from the posterior on weights.
    post_weight_rng, rng = jrnd.split(rng)
    def draw_weight_samples(mu, chol, rng, num_samples):
      Z = jrnd.normal(rng, (mu.shape[0], num_samples))
      return (mu[:,jnp.newaxis] + jnp.dot(chol.T, Z)).T
    
    # num_phases x num_post x num_rff
    post_weight_samples = jax.vmap( # vmap over phases
      draw_weight_samples,
      in_axes=(0, 0, 0, None),
    )(wt_post_means, wt_post_chols, jrnd.split(post_weight_rng, (num_phases,)),
      cfg.chase.posterior.num_samples,
    )

    ############################################################################
    # Get the posterior functions themselves.
    # num_post x num_candidates x num_phases <--- ugly 
    post_func_samples = jnp.einsum('ijk,lik->jli', post_weight_samples, predict_basis)

    ############################################################################
    # Compute the convex hull of the mins.
    def convex_hull(X, Y):
      min_Y = jnp.min(Y, axis=1)
      tight = lower_hull_points(X, min_Y)
      return tight[:,jnp.newaxis] * (min_Y[:,jnp.newaxis] == Y)

    # num_post x num_candidates x num_phases (boolean)    
    post_tight = jax.vmap(
      convex_hull,
      in_axes=(None,0),
    )(grid_x, post_func_samples)

    ############################################################################
    # Sample conditioned on the hull being the same.
    def hull_same_x(weights, target):
      funcs = jnp.einsum('ik,lik->li', weights, predict_basis)
      tight = convex_hull(grid_x, funcs)
      #jax.debug.print('off {}', jnp.argwhere(tight != target, size=2, fill_value=-1))
      #jax.debug.print('vals {}', funcs)
      return jnp.all(tight == target)
    
    def ess_step(rng, cur_wts, mean, chol_cov, target, phase_idx):
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

        # Compute new weights, updating only this phase.
        # Neither cur nor new have a mean offset.
        new_wts = cur.at[phase_idx].set(        
            jnp.cos(theta)*cur[phase_idx] + jnp.sin(theta)*nu
        ) + mean

        # Check the hull.
        done = jnp.logical_or(hull_same_x(new_wts, target), jnp.abs(upper-lower) < 1e-5) # MAKE PARAM

        # Shrink the bracket.
        upper = jnp.where(theta > 0, theta, upper)
        lower = jnp.where(theta < 0, theta, lower)

        return (rng, upper, lower, cur, nu, jnp.where(done, theta, jnp.nan))
      
      def draw_nu(rng, chol):
        Z = jrnd.normal(rng, (chol.shape[0],))
        return jnp.dot(chol.T, Z)

      nu_rng, rng = jrnd.split(rng)
      nu = draw_nu(nu_rng, chol_cov[phase_idx])

      #from apm.fake_lax import while_loop
      from jax.lax import while_loop
      _, _, _, _, _, theta = while_loop(
        _while_cond,
        _while_body,
        (rng, init_upper, init_lower, cur_wts-mean, nu, jnp.nan),
      )

      new_wts = jnp.cos(theta)*(cur_wts-mean) + jnp.sin(theta)*nu + mean
      
      return new_wts[phase_idx]

    # One approach: condition on which points are tight. In that case, we use
    # the posterior on weights to draw samples, and then effectively slice
    # off any part of the weight space that leads to a different set of points
    # being tight.  I think this is elegant, but it does not like to move very
    # much, even away from the tight points.

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

    # vmap over posterior samples to get distribution over weights that fixes
    # the tight points.
    # num_post x num_phases x num_rff
    # num_post x num_phases x num_rff x num_rff
    post_tight_means, post_tight_chols = jax.vmap(
      tight_params,
      in_axes=(0, 0, None, None),
    )(post_tight, post_func_samples, wt_post_means, wt_post_chols)

    def post_hull_sampler(rng, init_wts, ess_means, ess_chols, tight, num_samples):

      def _ess_scan(carry, _):
        rng, cur_wts = carry

        #new_wts = jax.vmap(ess_step, in_axes=(0, None, None, None, None, 0))(
        #  jrnd.split(step_rng, (num_phases,)), cur_wts, ess_means, ess_chols, tight,
        #  jnp.arange(num_phases))
        new_wts = []
        for phase_idx in range(num_phases):
          step_rng, rng = jrnd.split(rng)
          new_wts.append(ess_step(step_rng, cur_wts, ess_means, ess_chols, tight, phase_idx))
        new_wts = jnp.stack(new_wts)

        return (rng, new_wts), new_wts

      #from apm.fake_lax import scan
      from jax.lax import scan
      _, hull_wts = scan(
        _ess_scan,
        (rng, init_wts),
        jnp.arange(num_samples),
      )

      return hull_wts

    # vmap over posterior samples
    # num_post x num_samples x num_phases x num_rff
    post_hull_wts = jax.vmap(
      post_hull_sampler,
      in_axes=(0, 1, 0, 0, 0, None),
    )(jrnd.split(rng, post_tight_means.shape[0]), post_weight_samples, post_tight_means, post_tight_chols, post_tight, 50)
    #post_hull_sampler(rng, post_weight_samples[:,0,...], post_tight_means[0], post_tight_chols[0], post_tight[0], 10)
    print('post_hull_wts', post_hull_wts.shape)

    # Look at a sample that has the same hull.
    print('predict_basis', predict_basis.shape)
    post_hull_funcs = jnp.einsum('ijkl,mkl->ijkm', post_hull_wts, predict_basis)
    print('post_hull_funcs', post_hull_funcs.shape)

    anytight = jnp.any(post_tight, axis=2)

    # posterior index
    idx = 0

    ############################################################################
    # Make a plot to look at.
    plt.figure(figsize=(10,10))
    plt.plot(grid_x, post_func_samples[idx,:,0].T, color='blue', alpha=0.5)
    plt.plot(grid_x, post_func_samples[idx,:,1].T, color='red', alpha=0.5)
    plt.plot(data_x, data_y, 'o')

    plt.plot(grid_x, post_hull_funcs[idx,:,0,:].T, color='blue', alpha=0.1)
    plt.plot(grid_x, post_hull_funcs[idx,:,1,:].T, color='red', alpha=0.1)

    plt.plot(grid_x[anytight[idx]], 
             jnp.min(post_func_samples[idx,:,:], axis=1)[anytight[idx]],
             'kx',
    )
    plt.savefig("rff-ess-%d.png" % ii)

if __name__ == "__main__":
  main()

