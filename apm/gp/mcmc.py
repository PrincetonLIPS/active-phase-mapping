import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc

from jax.lax import while_loop, scan

from .marglike import log_marginal_likelihood_cholesky

PRNGKey = jax.Array
@jdc.pytree_dataclass
class SliceSamplerState:
  rng: PRNGKey
  X: jax.Array
  y: jax.Array
  noise: jax.Array
  cur_hypers: jax.Array
  box: jax.Array
  thresh: jnp.float32 = jnp.inf
  logprob: jnp.float32 = -jnp.inf
  iter: jnp.int32 = 0

def generate_slice_sampler(
    kernel_fn,
    ls_prior,
    amp_prior,
    num_samples,
    thinning,
    solve_method='cholesky',
  ):

  if solve_method != 'cholesky':
    raise ValueError("Only Cholesky decomposition is supported for now.")
  else:
    log_marginal_likelihood = log_marginal_likelihood_cholesky

  ##############################################################################
  # The conditional for determining when we've found a sample.
  # TODO: also stop when the box gets too small.
  # TODO: thread some error handling through here.
  def _shrink_while_cond(state: SliceSamplerState) -> jnp.bool:
    return jnp.logical_or(
        state.iter == 0,
        jnp.logical_or(
            jnp.isnan(state.logprob),
            state.thresh > state.logprob,
          )
      )
  ##############################################################################

  ##############################################################################
  # The main body for shrinking the box and sampling uniformly.
  def _shrink_while_body(state: SliceSamplerState) -> SliceSamplerState:
    
    # Sample from the current box.
    box_rng, rng = jrnd.split(state.rng)
    new_hypers = jrnd.uniform(
        box_rng,
        minval=state.box[:,0],
        maxval=state.box[:,1],
        shape=(state.box.shape[0],),
      )

    # Compute the log posterior at the new sample.
    logprob = log_marginal_likelihood(
        state.X, state.y, state.noise,
        new_hypers[:-1], new_hypers[-1],
        kernel_fn,
      )

    with jdc.copy_and_mutate(state) as new_state:

      # Shrink the box.
      new_state.box = jnp.vstack([
          jnp.where(new_hypers < state.cur_hypers, new_hypers, state.box[:,0]),
          jnp.where(new_hypers > state.cur_hypers, new_hypers, state.box[:,1]),
        ]).T

      # Update the hypers if we accept.
      new_state.cur_hypers = jnp.where(
          state.thresh < logprob,
          new_hypers,
          state.cur_hypers,
        )
      
      new_state.logprob = logprob
      new_state.rng = rng
      new_state.iter = state.iter + 1

    return new_state
  ##############################################################################

  ##############################################################################
  # The actual slice sampling step.
  @jax.jit
  def _slice_sample_step(state, _):

    # Get the slice.
    init_lp = log_marginal_likelihood(
      state.X,
      state.y,
      state.noise,
      state.cur_hypers[:-1],
      state.cur_hypers[-1],
      kernel_fn,
    )

    thresh_rng, rng = jrnd.split(state.rng)
    thresh = jnp.log(jax.random.uniform(thresh_rng)) + init_lp

    with jdc.copy_and_mutate(state) as new_state:
      new_state.thresh = thresh
      new_state.rng = rng
      new_state.iter = jnp.array(0)
      new_state.logprob = init_lp

      # The initial box comes from the prior bounds.
      new_state.box = jnp.array([
          *[[ls_prior[0], ls_prior[1]]]*state.X.shape[1],
          [amp_prior[0], amp_prior[1]],
        ])

    # Update the state until we accept.
    next_state = while_loop(_shrink_while_cond, _shrink_while_body, new_state)

    return next_state, None
  ##############################################################################

  ##############################################################################
  # This is actually get the post-thinning samples.
  # Nest it this way to hopefully use less memory than we would otherwise.
  def _outer_scan_body(init_state, _):
    final_state, _ = scan(
      _slice_sample_step,
      init_state, 
      None, 
      thinning,
    )
    return final_state, final_state
  ##############################################################################

  ##############################################################################
  # The function we're actually going to return, that can be vmapped over, etc.
  def _slice_sample_hypers(rng, X, y, noise, init_ls, init_amp):

    num_species = X.shape[1]

    # Combine the length scale and amplitude into a single vector.
    init_hypers = jnp.concatenate([init_ls, jnp.array([init_amp])])
    
    # Create a state object for the slice sampler.
    state = SliceSamplerState(
        rng = rng,
        X = X,
        y = y,
        noise = noise,
        cur_hypers = init_hypers,
        box = jnp.zeros((num_species+1,2)), # dummy initialization for shape
      )

    _, states = scan(_outer_scan_body, state, None, num_samples)

    hypers = states.cur_hypers

    # Split the parameters back out.
    lengthscales = hypers[:,:-1]
    amplitudes = hypers[:,-1]

    return lengthscales, amplitudes
  ##############################################################################
  
  return _slice_sample_hypers
