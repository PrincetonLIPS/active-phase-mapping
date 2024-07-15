import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import time

from .kernels import *
from .mcmc import generate_slice_sampler
from ..utils import multivariate_t_rvs

log = logging.getLogger()

class MultiGP:
  ''' Abstract class for implementing a multi-output Gaussian process. '''

  def __init__(self):
    pass

  def hyper_sample(self):
    pass



  # def __init__(self, rng, num_inputs, num_outputs, cfg):
  #   self.rng         = rng
  #   self.num_inputs  = num_inputs
  #   self.num_outputs = num_outputs
  #   self.kernel_fn   = globals()[cfg.kernel]
  #   self.ls_prior    = cfg.lengthscale_prior
  #   self.amp_prior   = cfg.amplitude_prior
  #   self.noise_prior = cfg.noise_prior
  #   self.num_rff     = cfg.num_rff

  #   self._hyper_init()
  #   self.sampler = generate_slice_sampler(cfg)

  # ##############################################################################
  # def _hyper_init(self):
  #   ''' Initialize the hyperparameters the first time using a prior draw. '''
  #   ls_rng, amp_rng, noise_rng, self.rng = jrnd.split(self.rng, 4)
  #   self.init_ls = jrnd.uniform(
  #       ls_rng, 
  #       (self.num_outputs, self.num_inputs,),
  #       minval=self.ls_prior[0], 
  #       maxval=self.ls_prior[1],
  #     )
  #   log.info("Initial lengthscales: %s" % self.init_ls)

  #   self.init_amp = jrnd.uniform(
  #       amp_rng, 
  #       (self.num_outputs,),
  #       minval=self.amp_prior[0],
  #       maxval=self.amp_prior[1],
  #     )
  #   log.info("Initial amplitudes: %s" % self.init_amp) 

  #   if self.noise_prior[0] == 0 and self.noise_prior[1] == 0:
  #     self.init_noise = jnp.zeros(self.num_outputs)
  #   else:
  #     self.init_noise = jrnd.uniform(
  #         noise_rng, 
  #         (self.num_outputs,),
  #         minval=self.noise_prior[0],
  #         maxval=self.noise_prior[1],
  #       )
  #   log.info("Initial noise: %s" % self.init_noise)

  # ##############################################################################
  # def hyper_sample(self, X, Y, num_samples, burnin, thinning):
  #   ''' Sample hyperparameters from the posterior. '''
  #   # FIXME: address nonzero noise.
  #   sampler_rng, self.rng = jrnd.split(self.rng)
  #   t0 = time.time()
  #   ls_samples, amp_samples = self.sampler(
  #       sampler_rng,
  #       X, Y,
  #       self.init_ls, self.init_amp,
  #       num_samples + burnin,
  #       thinning,
  #   )
  #   t1 = time.time()
  #   log.info("Hyperparameter sampling took %f seconds." % (t1-t0))


    
  #   return ls_samples, amp_samples
  
  # ##############################################################################
  # def get_rff_params(self):
  #   ''' Draw random Fourier features for the kernel. '''
  #   proj_rng, phase_rng, self.rng = jrnd.split(self.rng, 3)
  #   rff_projections = multivariate_t_rvs(
  #       proj_rng, 
  #       mu=jnp.zeros((self.num_inputs,)), 
  #       sigma=jnp.eye(self.num_inputs),
  #       df=5, # Matern 5/2
  #       shape=(self.num_outputs, self.num_rff),
  #     )   
  #   rff_phases = jrnd.uniform(
  #       phase_rng, 
  #       (self.num_outputs, self.num_rff),
  #       minval=0.0,
  #       maxval=2*jnp.pi,
  #     )
  #   return rff_projections, rff_phases

  # ##############################################################################
  # def draw_posterior_samples(
  #     self,
  #     train_X, train_Y,
  #     predict_X,
  #     hyper_samples,
  #     hyper_burnin,
  #     hyper_thinning,
  #     post_samples,
  #   ):

  #   # First, draw hyperparameter samples.
  #   ls_samples, amp_samples = self.hyper_sample(
  #       train_X, train_Y,
  #       hyper_samples, hyper_burnin, hyper_thinning,
  #     )
    
  #   # Get the RFF parameters.
  #   rff_projections, rff_phases = self.get_rff_params()

  #   # Evaluate the RFF at the training and prediction points.
  #   def rff_basis_function(x, rff_projections, rff_phases, ls, amp):
  #     rff = jnp.sqrt(2.0 / self.num_rff) * jnp.cos(
  #         jnp.dot(rff_projections, x/ls) + rff_phases
  #       )
  #     return amp * rff.squeeze()
    
  #   grid_basis = \
  #       jax.vmap( # vmap over phases
  #           jax.vmap( # vmap over hyperparameters
  #               jax.vmap( # vmap over the grid
  #                   rff_basis_function,
  #                   in_axes=(0, None, None, None, None,),
  #                 ),
  #               in_axes=(None, 0, 0, 0, 0,),
  #             ),
  #           in_axes=(None, 0, 0, 0, 0,),
  #         )(self.source.all_candidates, rff_projections, rff_phases, ls_samples, amp_samples)
    
  #   data_basis = \
  #       jax.vmap( # vmap over phases
  #           jax.vmap( # vmap over hyperparameters
  #               jax.vmap( # vmap over the grid
  #                   rff_basis_function,
  #                   in_axes=(0, None, None, None, None,),
  #                 ),
  #               in_axes=(None, 0, 0, 0, 0,),
  #             ),
  #           in_axes=(None, 0, 0, 0, 0,),
  #         )(X, rff_projections, rff_phases, ls_samples, amp_samples)