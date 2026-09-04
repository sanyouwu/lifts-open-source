import torch
import numpy as np
import random
from collections import deque

def generate_case4(num_sample_train=100, num_sample_test=100, p=3, T=20, d=None, 
                   obs_rate=0.5, coef_eps=1.0, seed=3, 
                   mask_noise_std=0.1, extra_steps=0, mm="ldm", df=10, obs_noise_std=0.3):
    """
    Case 4 (Final Revised v2): Linear AR(3) Latent Dynamics with Scaled Additive Noise.
    
    Observation Equation:
        X_t = F1 * Z_t + F2 * (exp(Z_t / 2) ⊙ (0.3 * η_{t,1})) + 0.3 * η_{t,2}
        
        Where:
        - η_{t,1} ~ t_10(0, I_d)  (Multiplicative noise source)
        - η_{t,2} ~ t_10(0, I_p)  (Additive noise source)
    
    Latent Dynamics (Linear AR(3)):
        Z_t = Phi1 * Z_{t-1} + Phi2 * Z_{t-2} + Phi3 * Z_{t-3} + 0.1 * v_t
        - v_t ~ N(0, I_d)
    """
    X_list = []
    Z_list = []
    mask_list = []
    X_complete = []
    
    X_list_test = []
    Z_list_test = []
    mask_list_test = []
    X_complete_test = []
    
    # Burn-in 
    burn_in = 100 
    
    T_total = T 
    T_total_test = T + extra_steps
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if d is None:
        d = p
        
    # ==========================================
    # 1. Mixing Matrices (F1, F2)
    # ==========================================
    F1_random = torch.randn(p, d)
    F2_random = torch.randn(p, d)
    
    # ==========================================
    # 2. AR(3) Transition Matrices (Phi1, 2, 3)
    # Scaling to ensure stability
    # ==========================================
    scale = 1.0 / (3.0 * np.sqrt(d)) 
    Phi1 = torch.randn(d, d) * scale
    Phi2 = torch.randn(d, d) * scale
    Phi3 = torch.randn(d, d) * scale
    Phi_list = [Phi1, Phi2, Phi3]
    
    # ==========================================
    # Generate Covariance Matrices Sigma_p and Sigma_d for correlated errors
    # Sigma = L @ L^T where L is lower triangular with positive diagonal
    # For Case 4: 
    #   η_{t,1} ~ t_10(0, Sigma_d)  (Multiplicative noise)
    #   η_{t,2} ~ t_10(0, Sigma_p)  (Additive noise)
    #   v_t ~ N(0, Sigma_d)         (Latent noise)
    # ==========================================
    # Sigma_d: (d, d) covariance for latent noise v_t and multiplicative noise eta_t1
    L_d = torch.randn(d, d)
    L_d = torch.tril(L_d)
    L_d = L_d + 0.5 * torch.eye(d)  # Ensure positive diagonal for PD
    Sigma_d = L_d @ L_d.T
    
    # Sigma_p: (p, p) covariance for additive observation noise eta_t2
    L_p = torch.randn(p, p)
    L_p = torch.tril(L_p)
    L_p = L_p + 0.5 * torch.eye(p)  # Ensure positive diagonal for PD
    Sigma_p = L_p @ L_p.T
    
    # Pre-compute Cholesky decomposition for sampling correlated noise
    L_d_chol = torch.linalg.cholesky(Sigma_d)  # For v_t ~ N(0, Sigma_d) and eta_t1 ~ t(0, Sigma_d)
    L_p_chol = torch.linalg.cholesky(Sigma_p)  # For eta_t2 ~ t(0, Sigma_p)

    # ==========================================
    # 3. Mask Parameters
    # ==========================================
    W_mask = torch.randn(p, d) * 1.0 
    if mm == "ldm":
        # 分段线性映射 for b_mask (扩展到极端 obs_rate)
        if obs_rate <= 0.1:
            b_value = 15.0 * obs_rate - 5.45
        elif obs_rate <= 0.2:
            b_value = 17.5 * obs_rate - 5.7
        else:
            b_value = 7.0 * obs_rate - 3.6
        b_mask = torch.ones(p) * b_value + torch.randn(p) * 0.2
    else:
        b_mask = torch.zeros(p)
    
    # 计算 mask_probs 的下界 (obs_rate/2) 以确保最小观测率
    mask_lower_bound = obs_rate / 2.0

    # ==========================================
    # Helpers
    # ==========================================
    def transition_step(history, phis, coef_eps):
        # history: [Z_{t-3}, Z_{t-2}, Z_{t-1}]
        term1 = phis[0] @ history[2] # Phi1 * Z_{t-1}
        term2 = phis[1] @ history[1] # Phi2 * Z_{t-2}
        term3 = phis[2] @ history[0] # Phi3 * Z_{t-3}
        
        z_det = term1 + term2 + term3
        # v_t ~ N(0, Sigma_d): use Cholesky to sample correlated noise
        v_t = L_d_chol @ torch.randn(d) * coef_eps * 0.1 
        return z_det + v_t

    def sample_t_distribution(shape, df, L_chol=None):
        """
        Sample from multivariate t-distribution with df degrees of freedom.
        t = N(0, Sigma) / sqrt(χ²(df) / df)
        
        If L_chol is provided, samples from t(0, Sigma) where Sigma = L_chol @ L_chol^T
        Otherwise samples from t(0, I).
        
        shape: tuple, output shape (d, T) or (p, T)
        df: int, degrees of freedom
        L_chol: optional Cholesky factor for covariance
        """
        if L_chol is not None:
            # Sample correlated normal: L @ N(0, I)
            dim = L_chol.shape[0]
            T_steps = shape[1]
            normal = L_chol @ torch.randn(dim, T_steps)
        else:
            normal = torch.randn(shape)
        chi2 = torch.sum(torch.randn(df, *shape) ** 2, dim=0)
        return normal / torch.sqrt(chi2 / df)

    # ==========================================
    # #### Train Generation
    # ==========================================
    for _ in range(num_sample_train):
        # 1. Init History
        z_hist = deque([torch.randn(d)*0.1 for _ in range(3)], maxlen=3)
        
        # 2. Burn-in
        for _ in range(burn_in):
            z_next = transition_step(list(z_hist), Phi_list, coef_eps)
            z_hist.append(z_next)
            
        # 3. Latent Z Generation
        Z = torch.zeros(d, T_total)
        for t in range(T_total):
            z_next = transition_step(list(z_hist), Phi_list, coef_eps)
            z_hist.append(z_next)
            Z[:, t] = z_next
        Z_list.append(Z)
        
        # 4. Observation X Generation
        # Term 1: Linear Projection F1 * Z
        term1 = F1_random @ Z  # (p, T)
        
        # Term 2: Nonlinear Multiplicative Noise
        # exp(Z/2) * (0.3 * eta1), eta1 ~ t_10(0, Sigma_d)
        eta1 = sample_t_distribution((d, T_total), df, L_d_chol) # (d, T), correlated t-distribution
        term2_inner = torch.exp(Z / 2) * (0.3 * eta1)
        term2 = F2_random @ term2_inner # (p, T)
        
        # Term 3: Additive Noise (Scaled by 0.3)
        # eta2 ~ t_10(0, Sigma_p)
        eta2 = sample_t_distribution((p, T_total), df, L_p_chol) # (p, T), correlated t-distribution
        term3 = 0.3 * eta2 # Applied scaling 0.3
        
        # Combine
        X = term1 + term2 + term3
        
        # Masking
        mask = torch.ones(p, T_total, dtype=torch.bool) 
        if obs_rate < 1.0:
            if mm == "mcar":
                for t in range(T):
                    mask[:, t] = torch.bernoulli(torch.ones(p) * obs_rate).bool()
            else: # LDM
                for t in range(T): 
                    z_t = Z[:, t]
                    mask_probs = torch.sigmoid(W_mask @ z_t + b_mask)
                    mask_probs = mask_lower_bound + (1 - mask_lower_bound) * mask_probs
                    mask[:, t] = torch.bernoulli(mask_probs).bool()
        
        if extra_steps > 0: mask[:, T:] = False

        mask_list.append(mask)
        X_obs = X.clone()
        X_obs[~mask] = torch.nan
        X_list.append(X_obs)
        X_complete.append(X)
        
    # ==========================================
    # #### Test Generation
    # ==========================================
    for _ in range(num_sample_test):
        z_hist = deque([torch.randn(d)*0.1 for _ in range(3)], maxlen=3)
        for _ in range(burn_in):
            z_next = transition_step(list(z_hist), Phi_list, coef_eps)
            z_hist.append(z_next)
            
        Z = torch.zeros(d, T_total_test)
        for t in range(T_total_test):
            z_next = transition_step(list(z_hist), Phi_list, coef_eps)
            z_hist.append(z_next)
            Z[:, t] = z_next
        Z_list_test.append(Z)
        
        # Observation X (Test)
        # eta1 ~ t_10(0, Sigma_d), eta2 ~ t_10(0, Sigma_p)
        eta1 = sample_t_distribution((d, T_total_test), df, L_d_chol)
        eta2 = sample_t_distribution((p, T_total_test), df, L_p_chol) # Dimension p, correlated
        
        term1 = F1_random @ Z
        term2 = F2_random @ (torch.exp(Z / 2) * (0.3 * eta1))
        term3 = 0.3 * eta2 # Scaled additive noise
        
        X = term1 + term2 + term3
        
        # Masking
        mask = torch.ones(p, T_total_test, dtype=torch.bool) 
        if obs_rate < 1.0:
            if mm == "mcar":
                for t in range(T_total_test):
                    mask[:, t] = torch.bernoulli(torch.ones(p) * obs_rate).bool()
            else:
                for t in range(T_total_test):
                    z_t = Z[:, t]
                    mask_probs = torch.sigmoid(W_mask @ z_t + b_mask)
                    mask_probs = mask_lower_bound + (1 - mask_lower_bound) * mask_probs
                    mask[:, t] = torch.bernoulli(mask_probs).bool()

        if extra_steps > 0: mask[:, T_total:] = False 

        mask_list_test.append(mask)
        X_obs = X.clone()
        X_obs[~mask] = torch.nan
        X_list_test.append(X_obs)
        X_complete_test.append(X)
        
    # Return Sigma_p and Sigma_d for compatibility (Case 4 has both)
    return [X_list, X_list_test], [mask_list, mask_list_test], [Z_list, Z_list_test], \
           [X_complete, X_complete_test], None, [F1_random, F2_random], [Phi_list, W_mask, b_mask, Sigma_p, Sigma_d]