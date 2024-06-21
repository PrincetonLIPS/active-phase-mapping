import jax
import jax.numpy as jnp
import jax.random as jrnd

from .kernels import Matern52

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
    chol_K: The Cholesky factor of the covariance matrix.
            [num_samples x num_phases x num_data x num_data]
  '''

  num_species = X.shape[1]
  num_phases = Y.shape[1]
  kernel_fn = globals()[cfg.gp.kernel]
  noise = 0.0 # Assumed
  jitter = cfg.gp.jitter

  apply_K = jax.jit(lambda X, ls, amp: amp * kernel_fn(X/ls) \
      + jnp.eye(X.shape[0]) * (noise+jitter))

  # We jointly sample the length scale and amplitude.

  # Get the initial box for each step based on the prior.
  init_box = jnp.array([
    *[[cfg.gp.lengthscale_prior[0], cfg.gp.lengthscale_prior[1]]]*num_species,
    [cfg.gp.amplitude_prior[0], cfg.gp.amplitude_prior[1]],
  ])

  # Combine the length scale and amplitude into a single vector.
  init_hypers = jnp.hstack([init_ls, init_amp[:,jnp.newaxis]]).T

  # Subtract off the means.
  Y = Y - jnp.mean(Y, axis=0)

  # The log posterior for a single phase.
  def log_posterior(hypers, y):
    ls  = hypers[:-1]
    amp = hypers[-1]
    K = apply_K(X, ls, amp)

    # Compute the log marginal likelihood.
    chol_K = jax.scipy.linalg.cho_factor(K)
    solved = jax.scipy.linalg.cho_solve(chol_K, y)
    log_marginal_likelihood = -0.5 * jnp.dot(y, solved) \
      - jnp.sum(jnp.log(jnp.diag(chol_K[0]))) \
      - 0.5 * X.shape[0] * jnp.log(2*jnp.pi)

    # The prior is uniform.
    return log_marginal_likelihood, chol_K[0]
  
  # The conditional for determining when we've found a sample.
  def _shrink_while_cond(state):
    thresh, cur_hypers, lp, box, rng, y, chol_K, iter = state
    return jnp.logical_or(iter == 0, jnp.logical_or(jnp.isnan(lp), thresh > lp))
  
  # The main body for shrinking the box and sampling uniformly.
  def _shrink_while_body(state):
    thresh, cur_hypers, lp, box, rng, y, chol_K, iter = state
    
    # Sample from the current box.
    box_rng, rng = jrnd.split(rng)
    new_hypers = jax.random.uniform(
      box_rng,
      minval=box[:,0],
      maxval=box[:,1],
      shape=(box.shape[0],),
    )

    # Compute the log posterior at the new sample.
    lp, chol_K = log_posterior(new_hypers, y)

    # Shrink the box.
    box = jnp.vstack([
      jnp.where(new_hypers < cur_hypers, new_hypers, box[:,0]),
      jnp.where(new_hypers > cur_hypers, new_hypers, box[:,1]),
    ]).T

    # Update the current hypers if we accept.
    cur_hypers = jnp.where(thresh < lp, new_hypers, cur_hypers)

    return thresh, cur_hypers, lp, box, rng, y, chol_K, iter+1

  # The main slice sampling step for a single phase.
  @jax.jit
  def _slice_sample_step(hypers, y, rng):
    thresh_rng, rng = jrnd.split(rng)

    # Get the slice.
    init_lp, chol_K = log_posterior(hypers, y)
    thresh = jnp.log(jax.random.uniform(thresh_rng)) + init_lp

    # Get the initial box for each step based on the prior.
    # Then sample and shrink until we accept.
    _, hypers, final_lp, _, _, _, chol_K, iters = jax.lax.while_loop(
      _shrink_while_cond,
      _shrink_while_body,
      (thresh, hypers, init_lp, init_box, rng, y, chol_K, jnp.array(0)),
    )
    return hypers, chol_K, iters, final_lp

  # We need this to get the type right for the scan.
  chol_K_placeholder = jnp.zeros((X.shape[0],X.shape[0]), dtype=jnp.float32)

  # A function we can loop, scan, or perhaps vmap over.
  def _sample_phase(init_hypers, y, rng):

    def _thin_scan_body(state, _):
      ''' Take a single slice sample step inside a scan.'''
      hypers, rng, chol_K = state
      sample_rng, rng = jrnd.split(rng)
      hypers, chol_K, _, _ = _slice_sample_step(hypers, y, sample_rng)
      return (hypers, rng, chol_K), None
  
    def _outer_scan_body(state, _):
      hypers, rng = state
      (hypers, rng, chol_K), _ = jax.lax.scan(
        _thin_scan_body, 
        (hypers, rng, chol_K_placeholder), 
        None, 
        thinning,
      )
      return (hypers, rng), (hypers, chol_K)
  
    _, (hypers, chol_K) = jax.lax.scan(
      _outer_scan_body, 
      (init_hypers, rng),
      None,
      num_samples,
    )

    return hypers, chol_K

  hypers, chol_K = jax.vmap(_sample_phase, in_axes=(1,1,0), out_axes=(1,1))(
    init_hypers, 
    Y, 
    jrnd.split(rng, num_phases),
  )

  # Split the parameters back out.
  lengthscales = hypers[:,:,:-1]
  amplitudes = hypers[:,:,-1]

  return lengthscales, amplitudes, chol_K
