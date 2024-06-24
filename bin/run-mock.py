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
  #chase.corner_init()
  #chase.select_next()


  # Plot the mockup function.
  fig = plt.figure()
  plt.plot(mockup.all_candidates[:,0], mockup.energies, "o")
  plt.savefig("mockup.pdf")

if __name__ == "__main__":
  main()

