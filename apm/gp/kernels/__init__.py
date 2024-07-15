from .matern52 import Matern52

def get_kernel(name: str):
  ''' Get a kernel by name.

      Args:
        name: The name of the kernel.

      Returns:
        The kernel.
  '''
  if name.lower() == 'matern52':
    return Matern52()
  else:
    raise ValueError(f'Unknown kernel: {name}')
  