# This is not a generic implementation of ESS, but a specific one for the
# multi-phase convex hull case.

from typing import Callable, Any
from jax.typing import ArrayLike
from jax import Array
from ..types import PRNGKey

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax_dataclasses as jdc

from ..hull import lower_hull_points
from ..chase import weight_posterior_cholesky # refactor

@jdc.pytree_dataclass
class HullESSSamplerState:
  rng:           PRNGKey
  phase_idx:     int
  theta:         float
  lower:         float
  upper:         float
  tight:         Array           # (num_phases, num_candidates)
  cur_weights:   Array           # (num_phases, num_rff)
  prop_weights:  Array           # (num_rff,)
  means:         Array           # (num_phases, num_rff)
  chol_covs:     Array           # (num_phases, num_rff, num_rff)
  iters:         Array           # (num_phases,)
  done:          bool = False

def generate_ess_sampler(
    cand_X:     ArrayLike, # num_candidates x num_species [float32]
    cand_basis: ArrayLike, # num_phases x num_candidates x num_rff [float32]
  ): # FIXME return type

  num_phases = cand_basis.shape[0]
  num_candidates = cand_basis.shape[1]
  num_rff = cand_basis.shape[2]

  def _convex_hull(
      Y: ArrayLike, # num_phases x num_candidates [float32]
    ) -> Array: # num_phases x num_candidates [bool]

    min_Y = jnp.min(Y, axis=0)
    tight = lower_hull_points(cand_X, min_Y)
    assert tight.shape == (cand_X.shape[0],)

    return tight[jnp.newaxis,:] * (min_Y[jnp.newaxis,:] == Y)

  def _ess_step_while_cond(state):
    return ~state.done

  def _ess_step_while_body(state):
    theta_rng, rng = jrnd.split(state.rng)

    # Sample uniformly in the interval.
    theta = jrnd.uniform(theta_rng, minval=state.lower, maxval=state.upper)

    # Compute the new weights.
    new_weights = state.cur_weights.at[state.phase_idx].set(
        jnp.cos(theta)*state.cur_weights[state.phase_idx] 
        + jnp.sin(theta)*state.prop_weights
      ) + state.means
    
    # Compute the functions.
    new_funcs = jnp.einsum(
        'ij,ikj->ik',
        new_weights,  # num_phases x num_rff
        cand_basis,   # num_phases x num_candidates x num_rff
      )
    assert new_funcs.shape == (num_phases, num_candidates)

    # Compute the convex hull.
    new_tight = _convex_hull(new_funcs)
    assert new_tight.shape == (num_phases, num_candidates)

    # Check if we are tight at the same locations.
    done = jnp.all(state.tight == new_tight)

    # Shrink the interval.  If we're done, we'll just ignore this.
    upper = jnp.where(theta > 0, theta, state.upper)
    lower = jnp.where(theta < 0, theta, state.lower)

    # Update the state.
    with jdc.copy_and_mutate(state) as new_state:
      new_state.rng = rng
      new_state.done = done
      new_state.theta = theta
      new_state.iters = state.iters.at[state.phase_idx].add(1)
      new_state.lower = lower
      new_state.upper = upper

    return new_state

  def _ess_phase_scan(
      state: HullESSSamplerState,
      phase_idx: int,
    ) -> HullESSSamplerState:
    # Update one phase.

    prop_rng, interval_rng, rng = jrnd.split(state.rng, 3)

    # Set up the state for the next phase.
    with jdc.copy_and_mutate(state) as new_state:
      new_state.rng = rng
      new_state.phase_idx = phase_idx
      new_state.done = jnp.array(False)
      new_state.iters = state.iters.at[phase_idx].set(0)

      # New proposed weights for a single phase.
      Z = jrnd.normal(
          prop_rng, 
          shape=state.cur_weights[phase_idx].shape,
        )
      new_state.prop_weights = new_state.chol_covs[phase_idx].T @ Z
      
      # Initial interval.
      new_state.upper = jrnd.uniform(interval_rng, minval=0.0, maxval=2*jnp.pi)
      new_state.lower = new_state.upper - 2*jnp.pi

    # Shrink the slice until we accept.
    new_state = jax.lax.while_loop(
      _ess_step_while_cond,
      _ess_step_while_body,
      new_state,
    )

    # Update current state using the final theta.
    with jdc.copy_and_mutate(new_state) as new_state:
      new_state.cur_weights = state.cur_weights.at[phase_idx].set(
          jnp.cos(new_state.theta)*state.cur_weights[phase_idx] 
          + jnp.sin(new_state.theta)*new_state.prop_weights
       )

    return new_state, None
  
  def _ess_scan(state: HullESSSamplerState, _) -> HullESSSamplerState:

    # Scan over phases.
    new_state, _ = jax.lax.scan(
      _ess_phase_scan,
      state,
      jnp.arange(state.cur_weights.shape[0]),
    )

    # Ultimately we want to return the samples as functions, not weights.
    funcs = jnp.einsum(
        'ij,ikj->ik',
        new_state.cur_weights + new_state.means, # num_phases x num_rff
        cand_basis,            # num_phases x num_candidates x num_rff
      )

    return new_state, funcs

  def _ess_sampler(
      rng:           PRNGKey,
      init_weights:  ArrayLike, # (num_phases, num_rff)
      post_mean:     ArrayLike, # (num_phases, num_rff)
      post_chol_cov: ArrayLike, # (num_phases, num_rff, num_rff)
      num_samples:   int,
      jitter:        float,
      noise_missing: float,
    ) -> Array:

    assert init_weights.shape == (num_phases, num_rff)
    assert post_mean.shape == (num_phases, num_rff)
    assert post_chol_cov.shape == (num_phases, num_rff, num_rff)

    init_funcs = jnp.einsum(
        'ij,ikj->ik',
        init_weights, # num_phases x num_rff
        cand_basis,   # num_phases x num_candidates x num_rff
      )
    assert init_funcs.shape == (num_phases, num_candidates)

    # Compute the convex hulk of the initial state.
    init_hull = _convex_hull(init_funcs)
    assert init_hull.shape == (num_phases, num_candidates)

    # Set the noise to be high for non-tight points.
    # This has the effect of treating the tight points as observed data.
    tight_noise = jnp.where(init_hull, jitter, noise_missing)

    # Compute the posterior for each phase.
    tight_means, tight_chol_covs = jax.vmap(
        weight_posterior_cholesky,
        in_axes=(0, 0, 0, 0, 0,),
      )(
        cand_basis,    # num_phases x num_candidates x num_rff
        init_funcs,    # num_phases x num_candidates
        tight_noise,   # num_phases x num_candidates
        post_mean,     # num_phases x num_rff
        post_chol_cov, # num_phases x num_rff x num_rff
      )
    assert tight_means.shape == (num_phases, num_rff)
    assert tight_chol_covs.shape == (num_phases, num_rff, num_rff)

    # Initialize the state.
    state = HullESSSamplerState(
        rng          = rng,
        theta        = 0.0, # dummy
        lower        = 0.0, # dummy
        upper        = 0.0, # dummy
        tight        = _convex_hull(init_funcs), # target hull
        cur_weights  = init_weights - tight_means, # IMPORTANT: removing mean
        prop_weights = jnp.zeros(init_weights.shape[1]), # dummy
        means        = tight_means,
        chol_covs    = tight_chol_covs,
        iters        = jnp.zeros(init_weights.shape[0]),
        done         = False,
        phase_idx    = 0,
    )

    # Scan over samples.
    _, samples = jax.lax.scan(
      _ess_scan,
      state,
      jnp.arange(num_samples),
    )

    # Make the sample dimension last, for consistency.
    return samples.transpose(1, 2, 0)

  return _ess_sampler