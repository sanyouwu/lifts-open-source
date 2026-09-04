import torch
import numpy as np
import random

def max_eig(A):
    """Helper function to compute the maximum eigenvalue magnitude."""
    return torch.max(torch.abs(torch.linalg.eigvals(A))).item()

def generate_case1(num_sample_train=100, num_sample_test=100, p=3, T=20, d=None, obs_rate=0.5, coef_eps=0.1, seed=3, 
                 mask_noise_std=0.1, extra_steps=0, mm="ldm"):
    """
    修改版：通过设计 Block-Diagonal 的 phi 和 稀疏的 F，使不同 Feature 差异显著。
    
    Args:
        extra_steps: int, 额外生成的时间步数（用于验证 forecasting）
                     如果 extra_steps=1，则生成 T+1 长度的数据，
                     前 T 步用于训练/条件，最后 1 步用于验证 forecasting
        mm: str, missing mechanism. "mcar" for Missing Completely At Random,
            "ldm" for Latent Dependent Missingness (MNAR).
        obs_rate: float, target observation rate. 
                  For MCAR: directly used as Bernoulli probability.
                  For LDM: used to set b_mask bias (approximate mapping).
    """
    X_list = []
    Z_list = []
    mask_list = []
    X_complete = []
    X_list_test = []
    Z_list_test = []
    mask_list_test = []
    X_complete_test = []
    burn_in = 100 # 稍微增加 burn_in 让震荡稳定
    
    # 实际生成的时间长度
    T_total = T 
    T_total_test = T + extra_steps
    
    # 设置种子
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if d is None:
        d = p
    phi = torch.randn(d, d) + 1.5 * torch.eye(d)  # 初始随机矩阵
    maxeig = max_eig(phi)
    if maxeig > 1.0:
        phi = phi / maxeig * 0.95

    # ==========================================
    # Generate Covariance Matrices Sigma_p and Sigma_d for correlated errors
    # Use a more numerically stable approach: Sigma = (1-rho)*I + rho*L@L^T
    # This ensures diagonal dominance and better conditioning
    # ==========================================
    rho = 0.3  # Correlation strength (0 = identity, 1 = fully correlated)
    
    # Sigma_d: (d, d) covariance for latent noise v_t
    L_d = torch.randn(d, d) * 0.5  # Scale down off-diagonal elements
    L_d = torch.tril(L_d)
    L_d.diagonal().add_(1.0)  # Ensure diagonal >= 1.0
    corr_d = L_d @ L_d.T
    # Normalize to correlation matrix then blend with identity
    D_d = torch.diag(1.0 / torch.sqrt(torch.diag(corr_d)))
    corr_d = D_d @ corr_d @ D_d  # Now a correlation matrix with 1s on diagonal
    Sigma_d = (1 - rho) * torch.eye(d) + rho * corr_d
    # Add small jitter for numerical stability
    Sigma_d = Sigma_d + 1e-4 * torch.eye(d)
    
    # Sigma_p: (p, p) covariance for observation noise r_t  
    L_p = torch.randn(p, p) * 0.5  # Scale down off-diagonal elements
    L_p = torch.tril(L_p)
    L_p.diagonal().add_(1.0)  # Ensure diagonal >= 1.0
    corr_p = L_p @ L_p.T
    # Normalize to correlation matrix then blend with identity
    D_p = torch.diag(1.0 / torch.sqrt(torch.diag(corr_p)))
    corr_p = D_p @ corr_p @ D_p  # Now a correlation matrix with 1s on diagonal
    Sigma_p = (1 - rho) * torch.eye(p) + rho * corr_p
    # Add small jitter for numerical stability
    Sigma_p = Sigma_p + 1e-4 * torch.eye(p)
    
    # Scale factor for observation noise (psi is kept for backward compatibility)
    psi_scale = 0.3
    psi = torch.eye(p) * psi_scale  # Legacy scalar version

    # ==========================================
    # 修改 2: 设计 F_random (Sparse Matrix)
    # 目的：让每个 x 只观测到特定的 z，防止"大杂烩"
    # ==========================================
    if d is None:
        F_random = torch.eye(d)
    else:
        # 1. 先生成标准正态分布
        # F_random = torch.randn(p, d) *0.3
        F_random = torch.randn(p, d) 

    W_mask = torch.randn(p, d) * 1.0 
    # ==========================================
    # b_mask 根据 obs_rate 设置 (仅对 LDM 有效)
    # 经验映射: obs_rate -> b_mask (扩展到极端 obs_rate)
    #   0.05 -> -4.7
    #   0.1  -> -3.95
    #   0.2  -> -2.2
    #   0.4  -> -0.5
    #   0.5  -> 0.1
    #   0.6  -> 0.5
    #   0.8  -> 2.0
    # 使用分段线性映射
    # ==========================================
    if mm == "ldm":
        if obs_rate <= 0.1:
            # 极端低obs_rate: 从 (0.05, -4.7) 和 (0.1, -3.95) 拟合
            # b ≈ 15 * obs_rate - 5.45
            b_value = 15.0 * obs_rate - 5.45
        elif obs_rate <= 0.2:
            # 从 (0.1, -3.95) 和 (0.2, -2.2) 拟合
            # b ≈ 17.5 * obs_rate - 5.7
            b_value = 17.5 * obs_rate - 5.7
        else:
            # 原有映射: 从 (0.2, -2.2) 和 (0.8, 2.0) 拟合: b ≈ 7 * obs_rate - 3.6
            b_value = 7.0 * obs_rate - 3.6
        b_mask = torch.ones(p) * b_value
        # 加一点随机扰动，让不同维度基础概率略有不同
        b_mask = b_mask + torch.randn(p) * 0.2
    else:
        # MCAR: b_mask 不使用，但仍需定义以兼容返回格式
        b_mask = torch.zeros(p)
    
    # 计算 mask_probs 的下界 (obs_rate/2) 以确保最小观测率
    mask_lower_bound = obs_rate / 2.0
    # ==========================================
    #### train set
    # ==========================================
    # Pre-compute Cholesky decomposition for sampling correlated noise
    L_d_chol = torch.linalg.cholesky(Sigma_d)  # For v_t ~ N(0, Sigma_d)
    L_p_chol = torch.linalg.cholesky(Sigma_p)  # For r_t ~ N(0, Sigma_p)
    
    for _ in range(num_sample_train):
        Z = torch.zeros(d, T_total+burn_in)
        Z[:, 0] = torch.randn(d)

        for t in range(1, T_total+ burn_in):
            # v_t ~ N(0, Sigma_d): use Cholesky to sample correlated noise
            v_t = L_d_chol @ torch.randn(d)
            Z[:, t] = phi @ Z[:, t-1] + coef_eps * v_t 
        Z = Z[:, burn_in:] 
        Z_list.append(Z)
        
        # Generate X with correlated observation noise
        # r_t ~ N(0, Sigma_p): sample correlated noise for each time step
        # X_t = F @ Z_t + 0.3 * r_t
        r_noise = L_p_chol @ torch.randn(p, T_total)  # (p, T_total)
        X = F_random @ Z + psi_scale * r_noise
        
        # Mask Generation 
        # 注意：mask 只对前 T 步生成（extra_steps 部分无 mask，因为是 forecasting 目标）
        mask = torch.ones(p, T_total, dtype=torch.bool) 
        
        if obs_rate >= 1.0:
            # Complete data: all observed, mask stays all True
            pass
        elif mm == "mcar":
            # MCAR: Missing Completely At Random
            for t in range(T):
                mask_probs_t = torch.ones(p) * obs_rate
                mask_t = torch.bernoulli(mask_probs_t).bool()
                mask[:, t] = mask_t
        else:
            # LDM: Latent Dependent Missingness (MNAR)
            for t in range(T):  # 只对前 T 步生成 mask
                # Get Z at time t: (d,)
                z_t = Z[:, t]
                mask_logits_t = W_mask @ z_t # (p,)
                mask_logits_t = mask_logits_t + b_mask # Add bias: (p,)
                mask_probs_t = torch.sigmoid(mask_logits_t) # (p,), values in [0, 1]
                mask_probs_t = mask_lower_bound + (1 - mask_lower_bound) * mask_probs_t
                mask_t = torch.bernoulli(mask_probs_t).bool() # (p,), values 0 or 1
                mask[:, t] = mask_t # Assign to mask matrix
        
        # 对于 extra_steps 部分，mask 设为全 0（全 missing），表示这些是 forecasting 目标
        if extra_steps > 0:
            mask[:, T:] = False  # T+1, T+2, ... 时刻全部 missing

        mask_list.append(mask)

        X_obs = X.clone()
        X_obs[~mask] = torch.nan
        X_list.append(X_obs)
        X_complete.append(X)
        
        
    ############### test set ##################
    ############### test set ##################
    ############### test set ##################
    ############### test set ##################
    ############### test set ##################
    for _ in range(num_sample_test):
        Z = torch.zeros(d, T_total_test+burn_in)
        Z[:, 0] = torch.randn(d)

        for t in range(1, T_total_test+ burn_in):
            # v_t ~ N(0, Sigma_d): use Cholesky to sample correlated noise
            v_t = L_d_chol @ torch.randn(d)
            Z[:, t] = phi @ Z[:, t-1] + coef_eps * v_t 
        Z = Z[:, burn_in:] 
        Z_list_test.append(Z)
        
        # Generate X with correlated observation noise
        r_noise = L_p_chol @ torch.randn(p, T_total_test)  # (p, T_total_test)
        X = F_random @ Z + psi_scale * r_noise
        
        # Mask Generation 
        mask = torch.ones(p, T_total_test, dtype=torch.bool) 
        
        if obs_rate >= 1.0:
            # Complete data: all observed, mask stays all True
            pass
        elif mm == "mcar":
            # MCAR: Missing Completely At Random
            for t in range(T_total_test):
                mask_probs_t = torch.ones(p) * obs_rate
                mask_t = torch.bernoulli(mask_probs_t).bool()
                mask[:, t] = mask_t
        else:
            # LDM: Latent Dependent Missingness (MNAR)
            for t in range(T_total_test):
                z_t = Z[:, t]
                mask_logits_t = W_mask @ z_t # (p,)
                mask_logits_t = mask_logits_t + b_mask # Add bias: (p,)
                mask_probs_t = torch.sigmoid(mask_logits_t) # (p,), values in [0, 1]
                mask_probs_t = mask_lower_bound + (1 - mask_lower_bound) * mask_probs_t
                mask_t = torch.bernoulli(mask_probs_t).bool() # (p,), values 0 or 1
                mask[:, t] = mask_t # Assign to mask matrix
        # 对于 extra_steps 部分，mask 设为全 0（全 missing），表示这些是 forecasting 目标
        if extra_steps > 0:
            mask[:, T_total:] = False  # T+1, T+2, ... 时刻全部 missing

        mask_list_test.append(mask)

        X_obs = X.clone()
        X_obs[~mask] = torch.nan
        X_list_test.append(X_obs)
        X_complete_test.append(X)
    # Return Sigma_p and Sigma_d for Kalman/RTS to use correct covariances
    # psi is kept for backward compatibility, but Sigma_p/Sigma_d are the true covariances
    return [X_list,X_list_test], [mask_list, mask_list_test], [Z_list, Z_list_test],[X_complete, X_complete_test], phi, F_random,[psi,W_mask, b_mask, Sigma_p, Sigma_d]