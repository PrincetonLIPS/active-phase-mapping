import jax
import jax.numpy  as jnp
import jax.random as jrnd
import matplotlib.pyplot as plt

def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, jnp.stack(ys)

def elliptical_slice(x0, log_lh_func, chol, num_samples, rng_key):

  @jax.jit
  def ess_step_condfun(state):
    x, new_x, nu, thresh, lower, upper, rng_key = state
    llh = log_lh_func(new_x)
    return log_lh_func(new_x) < thresh

  @jax.jit
  def ess_step_bodyfun(state):
    x, new_x, nu, thresh, lower, upper, rng_key = state
    theta_rng, rng_key = jrnd.split(rng_key, 2)
    theta = jrnd.uniform(theta_rng, minval=lower, maxval=upper)
    new_x = x*jnp.cos(theta) + nu*jnp.sin(theta)
    lower, upper = jax.lax.cond(theta < 0, lambda : (theta, upper), lambda : (lower, theta))
    return x, new_x, nu, thresh, lower, upper, rng_key

  @jax.jit
  def ess_step(x, rng_key):
    nu_rng, u_rng, theta_rng, rng_key = jrnd.split(rng_key, 4)

    # Determine slice height.
    u = jrnd.uniform(u_rng)
    thresh = log_lh_func(x) + jnp.log(u)

    # Get initial bracket.
    theta = jrnd.uniform(theta_rng, minval=0, maxval=2*jnp.pi)
    upper = theta
    lower = theta - 2*jnp.pi

    # Construct ellipse.
    nu = chol.T @ jrnd.normal(nu_rng, shape=x.shape)
    new_x = x*jnp.cos(theta) + nu*jnp.sin(theta)

    _, new_x, _, _, _, _, _ = jax.lax.while_loop(
      ess_step_condfun,
      ess_step_bodyfun,
      (x, new_x, nu, thresh, lower, upper, rng_key)
    )
    return new_x

  @jax.jit
  def scanfunc(state, xs):
    x, rng_key = state
    step_key, rng_key = jrnd.split(rng_key, 2)
    x = ess_step(x, step_key)
    return (x, rng_key), x

  _, samples = jax.lax.scan(scanfunc, (x0, rng_key), None, num_samples)

  return samples

rng_key = jrnd.PRNGKey(1)

S = jnp.array([
  [1.0, -1.2],
  [-1.2, 2.0],
])
import numpy.linalg as npla
cS = npla.cholesky(S)

samples = elliptical_slice(jnp.ones(2), lambda x: jnp.where(x[0]>0, 0, -jnp.inf), cS, 50000, rng_key)
print(jnp.mean(samples, axis=0))
print(jnp.cov(samples.T))
plt.plot(samples[:,0], samples[:,1], '.')
plt.gca().set_aspect('equal')
plt.show()
