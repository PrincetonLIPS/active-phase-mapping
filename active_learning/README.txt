Active learning for convex hulls. 

cal.py  -- General active learning procedure used in the paper. 
base.py -- Baseline active learning procedure--not hull aware. 
utils.py -- Lots of miscellaneous functions.
gp_model.py -- script detailing Gaussian Process regression model used in both policies. 
base_policy.py -- functions used in implementing active learning for baseline policy.
mpi.py -- Cal is written to be parallelized across cores using mpi. mpi.py provides function for parallelization of EIG calculation.
fps.py -- script for farthest point sampling.
