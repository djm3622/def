import torch
import scipy as sp
import numpy as np
import pylops

# TODO
# implement traditional SV-based forecasting
# nonlinear model M, tangent linear model J(M) [get this from torch.autograd]
# perturb along singular vectors and transform back

def etkf_perturb(x0, model, n_ensembles=20, inflation=1.1):
    """
    x0: Initial condition tensor (shape: [state_dim])
    model: Pretrained NN deterministic model
    Returns: Perturbed ICs (shape: [n_ensembles, state_dim])
    """
    state_dim = x0.shape[0]
    
    # 1. Generate initial perturbations (e.g., Gaussian)
    perturbations = torch.randn(n_ensembles, state_dim) * 0.01  # Small noise
    
    # 2. Compute ensemble forecasts
    with torch.no_grad():
        forecasts = torch.stack([model(x0 + pert) for pert in perturbations])
    
    # 3. Compute ensemble mean and anomalies
    ens_mean = forecasts.mean(dim=0)
    anomalies = forecasts - ens_mean  # Shape: [n_ens, state_dim]
    
    # 4. ETKF transform (simplified, no observations)
    # Covariance: (n_ens, n_ens) matrix
    cov = anomalies @ anomalies.T / (n_ensembles - 1)
    cov_inv = torch.linalg.pinv(cov)
    
    # Inflate covariance for numerical stability
    transform_matrix = torch.linalg.cholesky(inflation * cov_inv)
    
    # 5. Transform perturbations
    transformed_perturbations = perturbations @ transform_matrix
    
    return x0 + transformed_perturbations


def singular_vectors(model, x0, n_iters=5, sv_dim=10):
    """
    model: NN with .forward() method
    x0: Initial condition (shape: [state_dim])
    Returns: Top-10 singular vectors (shape: [sv_dim, state_dim])
    """
    x0.requires_grad_(True)
    
    # Forward pass to get forecast at t1
    y = model(x0)
    
    # Jacobian matrix (dy/dx0)
    J = []
    for i in range(y.shape[0]):
        grad = torch.autograd.grad(y[i], x0, retain_graph=True)[0]
        J.append(grad)
    J = torch.stack(J)  # Shape: [output_dim, state_dim]
    
    # Power iteration for top singular vectors
    v = torch.randn(sv_dim, J.shape[1])  # Initial random vectors
    for _ in range(n_iters):
        u = F.normalize(J @ v.T, dim=0)          # J * v
        v = F.normalize(J.T @ u, dim=0)          # J^T * u
    return v  # Right singular vectors (optimal perturbations)


