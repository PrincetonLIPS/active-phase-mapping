import hydra
import jax
import jax.numpy as jnp
import jax.random as jrnd
import logging
import matplotlib.pyplot as plt

#jax.config.update("jax_enable_x64", True)

from omegaconf import DictConfig

from apm.mockup import Mockup
from apm.chase import CHASE

log = logging.getLogger()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

  print(cfg)
  mockup = Mockup(cfg)

  chase = CHASE(cfg, mockup, jrnd.PRNGKey(1))
  chase.corner_init()
  chase.select_next()


  # Plot the mockup function.
  # fig = plt.figure()
  # ax = fig.add_subplot(projection='3d')
  # ax.scatter(
  #   mockup.all_candidates[:,0], 
  #   mockup.all_candidates[:,1], 
  #   mockup.energies.T,
  #   )
  # plt.savefig("mockup.pdf")

  #X = mockup.all_candidates[:100,:]
  #Y = mockup.energies[:100,:]

  #from apm.mcmc import slice_sample_hypers

  # Initialize hyperparameters with a random draw from the prior.
  # rng = jrnd.PRNGKey(1)
  # ls_rng, amp_rng, rng = jrnd.split(rng, 3)
  # init_ls = jrnd.uniform(
  #   ls_rng, 
  #   (cfg.mockup.num_phases, cfg.mockup.num_species,),
  #   minval=cfg.gp.lengthscale_prior[0], 
  #   maxval=cfg.gp.lengthscale_prior[1],
  # )
  # init_amp = jrnd.uniform(
  #   amp_rng, 
  #   (cfg.mockup.num_phases,),
  #   minval=cfg.gp.amplitude_prior[0],
  #   maxval=cfg.gp.amplitude_prior[1],
  # )
  # log.info("Initial lengthscales: %s" % init_ls)
  # log.info("Initial amplitudes: %s" % init_amp)

  # # Run the slice sampler.
  # ls_samples, amp_samples, chol_K = slice_sample_hypers(
  #   rng,
  #   X,
  #   Y,
  #   cfg,
  #   init_ls,
  #   init_amp,
  #   1000,
  #   thinning=1,
  # )
  # print(jnp.mean(ls_samples, axis=0))
  # print(jnp.mean(amp_samples, axis=0))

if __name__ == "__main__":
  main()

