import torch
import scipy as sp
import numpy as np
import pylops

# TODO
# implement traditional SV-based forecasting
# nonlinear model M, tangent linear model J(M) [get this from torch.autograd]
# perturb along singular vectors and transform back