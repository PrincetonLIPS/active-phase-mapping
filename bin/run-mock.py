import hydra
import logging

from omegaconf import DictConfig

from apm.mockup import Mockup

log = logging.getLogger()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

  print(cfg)
  mockup = Mockup(cfg)

  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(
    mockup.all_candidates[:,0], 
    mockup.all_candidates[:,1], 
    mockup.energies,
    )
  plt.savefig("mockup.pdf")

if __name__ == "__main__":
  main()

