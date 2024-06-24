import jax
import jax.numpy as jnp
import jax.random as jrnd

from jax.lax import while_loop, scan

from .gp.kernels import Matern52
from .gp import log_marginal_likelihood

def slice_sample_hypers(rng, X, Y, cfg, init_ls, init_amp, num_samples, thinning=1):
  ''' This function performs slice sampling of the hyperparameters of the 
  Gaussian process. Assumes zero noise.  The mean is subtracted from the
  outputs.  No stepping out is performed; instead the prior is assumed to be
  uniform on a box and that is taken to provide the bounds.  Hyperrectangle
  sampling is used rather than unidirectonal slice sampling.  

  Args:
    rng: The random number generator.
    X: The input data. [num_data x num_species]
    Y: The output data. [num_data x num_phases]
    cfg: The configuration dictionary.
    init_ls: The initial lengthscale. [num_phases x num_species]
    init_amp: The initial amplitude. [num_phases]
    num_samples: The number of samples to draw.
    thinning: The number of samples to throw away between kept samples.

  Returns:
    lengthscales: The sampled length scales.
            [num_samples x num_phases x num_species]
    amplitudes: The sampled amplitudes.
            [num_samples x num_phases]
  '''

  num_species = X.shape[1]
  num_phases = Y.shape[1]
  kernel_fn = globals()[cfg.gp.kernel]
  noise = 0.0 # Assumed
  jitter = cfg.gp.jitter
  noise_missing = cfg.gp.noise_missing

  # Get the initial box for each step based on the prior.
  init_box = jnp.array([
    *[[cfg.gp.lengthscale_prior[0], cfg.gp.lengthscale_prior[1]]]*num_species,
    [cfg.gp.amplitude_prior[0], cfg.gp.amplitude_prior[1]],
  ])

  # Combine the length scale and amplitude into a single vector.
  init_hypers = jnp.hstack([init_ls, init_amp[:,jnp.newaxis]]).T

  # FIXME: Subtract off the means.
  #Y = Y - jnp.mean(Y, axis=0)
  
  # The conditional for determining when we've found a sample.
  def _shrink_while_cond(state):
    thresh, cur_hypers, lp, box, rng, y, iter = state
    return jnp.logical_or(iter == 0, jnp.logical_or(jnp.isnan(lp), thresh > lp))
  
  # The main body for shrinking the box and sampling uniformly.
  def _shrink_while_body(state):
    thresh, cur_hypers, lp, box, rng, y, iter = state
    
    # Sample from the current box.
    box_rng, rng = jrnd.split(rng)
    new_hypers = jax.random.uniform(
      box_rng,
      minval=box[:,0],
      maxval=box[:,1],
      shape=(box.shape[0],),
    )

    # Compute the log posterior at the new sample.
    lp = log_marginal_likelihood(
      X, y, new_hypers[:-1], new_hypers[-1], kernel_fn, jitter, noise_missing,
    )

    # Shrink the box.
    box = jnp.vstack([
      jnp.where(new_hypers < cur_hypers, new_hypers, box[:,0]),
      jnp.where(new_hypers > cur_hypers, new_hypers, box[:,1]),
    ]).T

    # Update the current hypers if we accept.
    cur_hypers = jnp.where(thresh < lp, new_hypers, cur_hypers)

    return thresh, cur_hypers, lp, box, rng, y, iter+1

  # The main slice sampling step for a single phase.
  @jax.jit
  def _slice_sample_step(hypers, y, rng):
    thresh_rng, rng = jrnd.split(rng)

    # Get the slice.
    init_lp = log_marginal_likelihood(
      X, y, hypers[:-1], hypers[-1], kernel_fn, jitter, noise_missing,
    )
    thresh = jnp.log(jax.random.uniform(thresh_rng)) + init_lp

    # Get the initial box for each step based on the prior.
    # Then sample and shrink until we accept.
    _, hypers, final_lp, _, _, _, iters = while_loop(
      _shrink_while_cond,
      _shrink_while_body,
      (thresh, hypers, init_lp, init_box, rng, y, jnp.array(0)),
    )
    return hypers, iters, final_lp

  # A function we can loop, scan, or perhaps vmap over.
  def _sample_phase(init_hypers, y, rng):

    def _thin_scan_body(state, _):
      ''' Take a single slice sample step inside a scan.'''
      hypers, rng = state
      sample_rng, rng = jrnd.split(rng)
      hypers, _, _ = _slice_sample_step(hypers, y, sample_rng)
      return (hypers, rng), None
  
    def _outer_scan_body(state, _):
      hypers, rng = state
      (hypers, rng), _ = scan(
        _thin_scan_body, 
        (hypers, rng), 
        None, 
        thinning,
      )
      return (hypers, rng), (hypers,)
  
    _, (hypers,) = scan(
      _outer_scan_body, 
      (init_hypers, rng),
      None,
      num_samples,
    )

    return hypers

  hypers = jax.vmap(_sample_phase, in_axes=(1,1,0), out_axes=1)(
    init_hypers, 
    Y, 
    jrnd.split(rng, num_phases),
  )

  # Split the parameters back out.
  lengthscales = hypers[:,:,:-1]
  amplitudes = hypers[:,:,-1]

  return lengthscales, amplitudes
