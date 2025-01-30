import torch
import numpy as np
import scipy as sp


# TODO : init things correctly
def benchmark_perturbations(x0, model):
    # Generate perturbations
    diffusion_perturbations = # TODO
    etkf_perturbations = etkf_perturb(x0, model, n_ensembles=50)
    sv_perturbations = singular_vectors(model, x0).unsqueeze(0)
    
    # Run forecasts : correct the perturbations, i.e. time embeddings
    with torch.no_grad():
        diffusion_forecasts = [model(pert) for pert in diffusion_perturbations]
        etkf_forecasts = [model(pert) for pert in etkf_perturbations]
        sv_forecasts = [model(pert) for pert in sv_perturbations]
    
    # Compute metrics : compute the metrics
    pass