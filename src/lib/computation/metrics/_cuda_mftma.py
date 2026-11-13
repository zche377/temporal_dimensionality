'''
This is a PyTorch implementation of the analysis method developed by SueYeon Chung.
For further details, please see the following two papers:

Classification and Geometry of General Perceptual Manifolds (Phys. Rev. X 2018)
Separability and Geometry of Object Manifolds in Deep Neural Networks

This implementation supports GPU acceleration through CUDA.
'''

import torch
import torch.nn as nn
from torch.autograd import Function
import cvxopt
from cvxopt import matrix, solvers
from typing import Optional, Tuple, List
import warnings

# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1000000
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12

def get_device():
    """Helper function to get the device (CPU/CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(tensor):
    """Helper function to move tensor to available device."""
    return tensor.to(get_device())

def cuda_manifold_analysis_corr(XtotT: List[torch.Tensor], kappa: float, n_t: int, 
                          t_vecs: Optional[List[torch.Tensor]] = None, n_reps: int = 10) -> Tuple:
    """
    Carry out the analysis on multiple manifolds.

    Args:
        XtotT: List of 2D tensors of shape (N, P_i) where N is the dimensionality
               of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional list of 2D tensors of shape (Dm_i, n_t) containing gaussian vectors
        n_reps: Number of repetitions for optimization

    Returns:
        Tuple containing:
        - a_Mfull_vec: 1D tensor containing the capacity calculated from each manifold
        - R_M_vec: 1D tensor containing the calculated anchor radius of each manifold
        - D_M_vec: 1D tensor containing the calculated anchor dimension of each manifold
        - res_coeff0: Residual correlation
        - KK: Dimensionality of low rank structure
    """
    device = get_device()
    num_manifolds = len(XtotT)
    
    # Move all input tensors to device
    XtotT = [to_device(x) if not isinstance(x, torch.Tensor) else x.to(device) for x in XtotT]
    
    # Compute global mean
    Xori = torch.cat(XtotT, dim=1)
    X_origin = torch.mean(Xori, dim=1, keepdim=True)
    
    # Center the manifolds
    Xtot0 = [x - X_origin for x in XtotT]
    centers = torch.stack([torch.mean(x, dim=1) for x in XtotT], dim=1)
    center_mean = torch.mean(centers, dim=1, keepdim=True)
    
    # Center correlation analysis
    U, S, V = torch.svd(centers - center_mean)
    total = torch.cumsum(torch.square(S)/torch.sum(torch.square(S)), dim=0)
    maxK = torch.argmax(torch.where(total < 0.95, total, torch.zeros_like(total))) + 11
    
    # Compute low rank structure
    norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0 = fun_FA(
        centers, maxK.item(), 20000, n_reps)
    
    res_coeff_opt, KK = torch.min(res_coeff), torch.argmin(res_coeff) + 1
    
    # Compute projection vector
    V11 = torch.matmul(Proj, V1_mat[KK - 1])
    X_norms = []
    XtotInput = []
    
    for Xr in Xtot0:
        # Project manifold data
        Xr_ns = Xr - torch.matmul(V11, torch.matmul(V11.T, Xr))
        Xr0_ns = torch.mean(Xr_ns, dim=1)
        Xr0_ns_norm = torch.norm(Xr0_ns)
        X_norms.append(Xr0_ns_norm)
        XtotInput.append((Xr_ns - Xr0_ns.reshape(-1, 1))/Xr0_ns_norm)
    
    # Initialize result tensors
    a_Mfull_vec = torch.zeros(num_manifolds, device=device)
    R_M_vec = torch.zeros(num_manifolds, device=device)
    D_M_vec = torch.zeros(num_manifolds, device=device)
    
    # Process each manifold
    for i in range(num_manifolds):
        S_r = XtotInput[i]
        D, m = S_r.shape
        
        if D > m:
            Q, R = torch.qr(S_r)
            S_r = torch.matmul(Q.T, S_r)
            D, m = S_r.shape
            
        sD1 = torch.cat([S_r, torch.ones((1, m), device=device)], dim=0)
        
        if t_vecs is not None:
            t_vec = to_device(t_vecs[i]) if not isinstance(t_vecs[i], torch.Tensor) else t_vecs[i].to(device)
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t, t_vec=t_vec)
        else:
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t)
            
        a_Mfull_vec[i] = a_Mfull
        R_M_vec[i] = R_M
        D_M_vec[i] = D_M
        
    return a_Mfull_vec, R_M_vec, D_M_vec, res_coeff0, KK

def each_manifold_analysis_D1(sD1: torch.Tensor, kappa: float, n_t: int, 
                             eps: float = 1e-8, t_vec: Optional[torch.Tensor] = None) -> Tuple:
    """
    Compute manifold capacity, radius and dimension for a single manifold.
    
    Args:
        sD1: 2D tensor of shape (D+1, m) where m is number of manifold points
        kappa: Margin size (scalar)
        n_t: Number of randomly sampled vectors to use
        eps: Minimal distance
        t_vec: Optional 2D tensor of shape (D+1, m) containing sampled t vectors
        
    Returns:
        Tuple containing:
        - a_Mfull: Calculated capacity (scalar)
        - R_M: Calculated radius (scalar)
        - D_M: Calculated dimension (scalar)
    """
    device = get_device()
    D1, m = sD1.shape
    D = D1 - 1
    
    if t_vec is None:
        t_vec = torch.randn(D1, n_t, device=device)
    
    ss, gg = maxproj(t_vec, sD1)
    
    s_all = torch.empty((D1, n_t), device=device)
    f_all = torch.zeros(n_t, device=device)
    
    for i in range(n_t):
        t = t_vec[:, i:i+1]
        if gg[i] + kappa < 0:
            v_f = t
            s_f = ss[:, i:i+1]
        else:
            v_f, _, _, alpha, vminustsqk = minimize_vt_sq(t, sD1, kappa=kappa)
            f_all[i] = vminustsqk
            
            if torch.norm(v_f - t) < eps:
                v_f = t
                s_f = ss[:, i:i+1]
            else:
                scale = torch.sum(alpha)
                s_f = (t - v_f)/scale
                
        s_all[:, i] = s_f.squeeze()
    
    # Compute capacity
    max_ts = torch.maximum(torch.sum(t_vec * s_all, dim=0) + kappa, 
                          torch.zeros(n_t, device=device))
    s_sum = torch.sum(torch.square(s_all), dim=0)
    lamb = torch.where(s_sum > 0, max_ts/s_sum, torch.zeros_like(s_sum))
    slam = torch.square(lamb) * s_sum
    a_Mfull = 1/torch.mean(slam)
    
    # Compute R_M
    ds0 = s_all - torch.mean(s_all, dim=1, keepdim=True)
    ds = ds0[0:-1, :]/s_all[-1, :]
    ds_sq_sum = torch.sum(torch.square(ds), dim=0)
    R_M = torch.sqrt(torch.mean(ds_sq_sum))
    
    # Compute D_M
    t_norms = torch.sum(torch.square(t_vec[0:D, :]), dim=0, keepdim=True)
    t_hat_vec = t_vec[0:D, :]/torch.sqrt(t_norms)
    s_norms = torch.sum(torch.square(s_all[0:D, :]), dim=0, keepdim=True)
    s_hat_vec = s_all[0:D, :]/torch.sqrt(s_norms + 1e-12)
    ts_dot = torch.sum(t_hat_vec * s_hat_vec, dim=0)
    
    D_M = D * torch.square(torch.mean(ts_dot))
    
    return a_Mfull, R_M, D_M

def maxproj(t_vec: torch.Tensor, sD1: torch.Tensor, sc: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find points on manifold with largest projection onto each t vector.
    
    Args:
        t_vec: 2D tensor of shape (D+1, n_t)
        sD1: 2D tensor of shape (D+1, m)
        sc: Value for center dimension
        
    Returns:
        Tuple containing:
        - s0: 2D tensor of shape (D+1, n_t) with maximum projection points
        - gt: 1D tensor of shape (n_t) with maximum projection values
    """
    device = get_device()
    D1, n_t = t_vec.shape
    D = D1 - 1
    m = sD1.shape[1]
    
    s0 = torch.zeros((D1, n_t), device=device)
    gt = torch.zeros(n_t, device=device)
    
    for i in range(n_t):
        t = t_vec[:, i]
        max_S = torch.argmax(torch.matmul(t[0:D], sD1[0:D]))
        sr = sD1[0:D, max_S]
        s0[:, i] = torch.cat([sr, torch.tensor([sc], device=device)])
        gt[i] = torch.dot(t, s0[:, i])
        
    return s0, gt

def minimize_vt_sq(t: torch.Tensor, sD1: torch.Tensor, kappa: float = 0) -> Tuple:
    """
    Carry out constrained minimization using cvxopt.
    
    Args:
        t: Single T vector as 2D tensor (D+1, 1)
        sD1: 2D tensor (D+1, m)
        kappa: Size of margin
        
    Returns:
        Tuple containing optimization results
    """
    # Convert PyTorch tensors to numpy for cvxopt
    t_np = t.cpu().numpy()
    sD1_np = sD1.cpu().numpy()
    
    D1 = t_np.shape[0]
    m = sD1_np.shape[1]
    
    P = matrix(np.identity(D1))
    q = matrix(-t_np.astype(np.double))
    G = matrix(sD1_np.T.astype(np.double))
    h = matrix(-np.ones(m) * kappa)
    
    output = solvers.qp(P, q, G, h)
    
    # Convert results back to PyTorch tensors on appropriate device
    device = get_device()
    v_f = torch.tensor(np.array(output['x']), device=device)
    vt_f = torch.tensor(output['primal objective'], device=device)
    exitflag = torch.tensor(1 if output['status'] == 'optimal' else 0, device=device)
    alphar = torch.tensor(np.array(output['z']), device=device)
    normvt2 = torch.sum(torch.square(v_f - t))
    
    return v_f, vt_f, exitflag, alphar, normvt2

def fun_FA(centers: torch.Tensor, maxK: int, max_iter: int, n_repeats: int,
           s_all: Optional[List[torch.Tensor]] = None, verbose: bool = False) -> Tuple:
    """
    Extracts the low rank structure from the data given by centers.
    
    Args:
        centers: 2D tensor of shape (N, P)
        maxK: Maximum rank to consider
        max_iter: Maximum number of iterations
        n_repeats: Number of repetitions for stable solution
        s_all: Optional list of (P, 1) random normal vectors
        verbose: Whether to print progress
        
    Returns:
        Tuple containing optimization results
    """
    device = get_device()
    N, P = centers.shape
    
    # Center the data
    mean = torch.mean(centers.T, dim=0, keepdim=True)
    Xb = centers.T - mean
    xbnorm = torch.sqrt(torch.sum(torch.square(Xb), dim=1, keepdim=True))
    
    # Gram-Schmidt orthogonalization using current PyTorch linalg API
    Q, _ = torch.linalg.qr(Xb.T, mode='reduced')
    X = torch.matmul(Xb, Q[:, 0:P-1])
    
    # Store original data
    X0 = X.clone()
    xnorm = torch.sqrt(torch.sum(torch.square(X0), dim=1, keepdim=True))
    
    # Calculate correlations
    C0 = torch.matmul(X0, X0.T)/torch.matmul(xnorm, xnorm.T)
    res_coeff0 = (torch.sum(torch.abs(C0)) - P) * 1/(P * (P - 1))
    
    # Storage for results
    V1_mat = []
    C0_mat = []
    norm_coeff = []
    norm_coeff_vec = []
    res_coeff = []
    
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = 0
        
        for j in range(1, n_repeats + 1):
            # Sample random vector or use provided one
            if s_all is not None and len(s_all) >= i:
                s = s_all[i*j - 1].to(device)
            else:
                s = torch.randn(P, 1, device=device)
            
            # Create initial V
            sX = torch.matmul(s.T, X)
            if V1 is None:
                V0 = sX
            else:
                V0 = torch.cat([sX, V1.T], dim=0)
            V0, _ = torch.linalg.qr(V0.T, mode='reduced')  # (P-1, i)
            
            # Ensure V0 is properly shaped before optimization
            V0 = V0.view(V0.shape[0], -1)  # Reshape to ensure 2D
            
            # Optimize V using Stiefel manifold optimization
            V1tmp = stiefel_optimize(V0, X, max_iter)
            
            # Compute cost
            cost_after = compute_cost(V1tmp, X)
            
            # Verify orthogonality
            orth_error = torch.norm(torch.matmul(V1tmp.T, V1tmp) - 
                                  torch.eye(i, device=device), p='fro')
            # assert orth_error < 1e-10
            
            # Extract low rank structure
            X0 = X - torch.matmul(torch.matmul(X, V1tmp), V1tmp.T)
            
            # Compute stability
            denom = torch.sqrt(torch.sum(torch.square(X), dim=1))
            stability = torch.min(torch.sqrt(torch.sum(torch.square(X0), dim=1))/denom)
            
            if stability > best_stability:
                best_stability = stability
                best_V1 = V1tmp
                
            if n_repeats > 1 and verbose:
                print(f"Iteration {j}, cost={cost_after}, stability={stability}")
        
        V1 = best_V1
        
        # Extract low rank structure
        XV1 = torch.matmul(X, V1)
        X0 = X - torch.matmul(XV1, V1.T)
        
        # Compute current cost
        xnorm = torch.sqrt(torch.sum(torch.square(X0), dim=1, keepdim=True))
        C0 = torch.matmul(X0, X0.T)/torch.matmul(xnorm, xnorm.T)
        current_cost = (torch.sum(torch.abs(C0)) - P) * 1/(P * (P - 1))
        
        if verbose:
            print(f'K={i}, mean={current_cost}')
        
        # Store results
        V1_mat.append(V1)
        C0_mat.append(C0)
        norm_coeff.append((xnorm/xbnorm)[:, 0])
        norm_coeff_vec.append(torch.mean(xnorm/xbnorm))
        res_coeff.append(current_cost)
        
        # Check convergence
        if (i > 4 and 
            res_coeff[i-1] > res_coeff[i-2] and
            res_coeff[i-2] > res_coeff[i-3] and
            res_coeff[i-3] > res_coeff[i-4]):
            if verbose:
                print("Optimal K0 found")
            break
            
    return norm_coeff, norm_coeff_vec, Q[:, 0:P-1], V1_mat, res_coeff, res_coeff0

def stiefel_optimize(X0: torch.Tensor, A: torch.Tensor, max_iter: int) -> torch.Tensor:
    """
    Optimize on the Stiefel manifold using Riemannian gradient descent.
    
    Args:
        X0: Initial point on Stiefel manifold
        A: Data tensor
        max_iter: Maximum iterations
        
    Returns:
        Optimized point on Stiefel manifold
    """
    device = get_device()
    X = X0.clone()
    
    for _ in range(max_iter):
        # Compute gradient
        grad = compute_gradient(X, A)
        
        # Project gradient onto tangent space
        proj_grad = project_tangent(X, grad)
        
        # Perform line search
        alpha = 0.1
        X_new = retract(X, -alpha * proj_grad)
        
        # Check convergence
        if torch.norm(proj_grad) < 1e-6:
            break
            
        X = X_new
        
    return X

def compute_gradient(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of the cost function with correct broadcasting.
    
    Args:
        X: Current point on Stiefel manifold
        A: Data tensor
        
    Returns:
        Gradient tensor
    """
    P = A.shape[0]
    C = torch.matmul(A, A.T)
    c = torch.matmul(A, X)
    c0 = torch.diagonal(C).reshape(P, 1) - torch.sum(torch.square(c), dim=1, keepdim=True)
    
    # Reshape tensors for proper broadcasting
    X1 = A.view(P, 1, -1, 1)
    X2 = A.view(1, P, -1, 1)
    C1 = c.view(P, 1, 1, -1)
    C2 = c.view(1, P, 1, -1)
    
    # Compute correlation terms with proper broadcasting
    Cmcc = C - torch.matmul(c, c.T)
    c0_prod = torch.matmul(c0, c0.T)
    
    PF1 = (Cmcc/c0_prod).view(P, P, 1, 1)
    PF2 = (torch.square(Cmcc)/torch.square(c0_prod)).view(P, P, 1, 1)
    
    # Calculate gradient components with correct dimensions
    Gmni = torch.zeros_like(X1)
    Gmni = Gmni - PF1 * C1 * X1
    Gmni = Gmni - PF1 * C2 * X2
    Gmni = Gmni + PF2 * c0.view(P, 1, 1, 1) * C2 * X1
    Gmni = Gmni + PF2 * c0.T.view(1, P, 1, 1) * C1 * X2
    
    # Sum over the appropriate dimensions
    return torch.sum(Gmni, dim=[0, 1]).squeeze(-1)

def project_tangent(X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Project gradient onto tangent space of Stiefel manifold.
    
    Args:
        X: Point on Stiefel manifold (n x p matrix)
        G: Gradient tensor (n x p matrix)
        
    Returns:
        Projected gradient
    """
    # Ensure inputs are 2D matrices
    if X.dim() == 1:
        X = X.view(-1, 1)
    if G.dim() == 1:
        G = G.view(-1, 1)
        
    # Compute symmetric part using matrix multiplication
    sym = (torch.matmul(G.transpose(-2, -1), X) + 
           torch.matmul(X.transpose(-2, -1), G)) / 2
           
    return G - torch.matmul(X, sym)

def retract(X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Retract point back onto Stiefel manifold.
    
    Args:
        X: Current point
        G: Descent direction
        
    Returns:
        Retracted point on manifold
    """
    Q, R = torch.qr(X + G)
    return Q

def compute_cost(V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Compute the cost function value.
    
    Args:
        V: Current point on Stiefel manifold
        X: Data tensor
        
    Returns:
        Cost value
    """
    P = X.shape[0]
    C = torch.matmul(X, X.T)
    c = torch.matmul(X, V)
    c0 = torch.diagonal(C).reshape(P, 1) - torch.sum(torch.square(c), dim=1, keepdim=True)
    Fmn = torch.square(C - torch.matmul(c, c.T))/torch.matmul(c0, c0.T)
    return torch.sum(Fmn)/2