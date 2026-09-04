import torch
import numpy as np
import random

def generate_case3(num_sample_train=100, num_sample_test=100, p=3, T=20, d=None, 
                   obs_rate=0.5, coef_eps=1.0, seed=3, 
                   mask_noise_std=0.1, extra_steps=0, mm="ldm", df=10, obs_noise_std=0.3):
    """
    Case 3: Generalized Stochastic Volatility with Nonlinear Dynamics.
    
    Observation Equation:
        X_t = F * (exp(Z_t / 2) ⊙ η_t), where η_t ~ t_{df}(0, I_d)
    
    Latent Dynamics (element-wise):
        Z_{t,j} = tanh(0.5 * Z_{t-1,j} + 0.5) + v_{t,j}, where v_{t,j} ~ N(0, 1)
    
    Args:
        num_sample_train: int, number of training samples.
        num_sample_test: int, number of test samples.
        p: int, observation dimension.
        T: int, time series length.
        d: int, latent dimension. Default is p if None.
        obs_rate: float, target observation rate.
        coef_eps: float, standard deviation of process noise v_t. Default 1.0.
        seed: int, random seed.
        mask_noise_std: float, not used but kept for API compatibility.
        extra_steps: int, additional time steps for forecasting validation.
        mm: str, missing mechanism. "mcar" for Missing Completely At Random,
            "ldm" for Latent Dependent Missingness (MNAR).
        df: int, degrees of freedom for t-distribution. Default 10.
        obs_noise_std: float, observation noise standard deviation (eta scale). Default 0.3.
    """
    X_list = []
    Z_list = []
    mask_list = []
    X_complete = []
    
    X_list_test = []
    Z_list_test = []
    mask_list_test = []
    X_complete_test = []
    
    # Burn-in for the nonlinear dynamics to stabilize
    burn_in = 100 
    
    # 实际生成的时间长度
    T_total = T 
    T_total_test = T + extra_steps
    
    # 设置种子
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if d is None:
        d = p
        
    # ==========================================
    # 1. 定义 Mixing Matrix F 和 Intercept (用于观测方程)
    # Equation: X = F * (exp(Z/2) ⊙ η) + intercept
    # F 矩阵 i.i.d. 标准正态
    # ==========================================
    F_random = torch.randn(p, d)
    
    # Intercept 项: 每个 feature 有不同的 baseline
    # intercept ~ Uniform(-1, 1)
    intercept = torch.rand(p) * 2 - 1  # (p,)

    # psi 保留以兼容返回格式 (Case 3 没有加性观测噪声)
    psi = torch.eye(p) * 0.0
    
    # ==========================================
    # Generate Covariance Matrix Sigma_d for correlated errors
    # Sigma = L @ L^T where L is lower triangular with positive diagonal
    # For Case 3: η_t ~ t_10(0, Sigma_d)
    # ==========================================
    L_d = torch.randn(d, d)
    L_d = torch.tril(L_d)
    L_d = L_d + 0.5 * torch.eye(d)  # Ensure positive diagonal for PD
    Sigma_d = L_d @ L_d.T

    # ==========================================
    # 2. MNAR Mask 参数 (W_mask, b_mask)
    # 保持与 Case 1, 2 相同的 Missing Mechanism 逻辑
    # ==========================================
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
            b_value = 15.0 * obs_rate - 5.45
        elif obs_rate <= 0.2:
            # 从 (0.1, -3.95) 和 (0.2, -2.2) 拟合
            b_value = 17.5 * obs_rate - 5.7
        else:
            # 原有映射: b ≈ 7 * obs_rate - 3.6
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
    # 辅助函数: Nonlinear Transition (tanh-based)
    # Z_{t,j} = tanh(0.5 * Z_{t-1,j} + 0.5) + v_{t,j}
    # ==========================================
    def nonlinear_transition(z_prev, coef_eps):
        """
        Apply nonlinear transition: tanh(0.5 * z + 0.5) + 0.3 * noise
        z_prev: (d,) tensor
        Returns: (d,) tensor
        """
        z_det = torch.tanh(0.5 * z_prev + 0.5)
        v_t = torch.randn(d) * coef_eps * 0.3  # 缩小 process noise
        return z_det + v_t

    # ==========================================
    # 辅助函数: Generate t-distributed noise with covariance Sigma_d
    # η_t ~ t_{df}(0, Sigma_d)
    # ==========================================
    # Pre-compute Cholesky decomposition for correlated t-distribution sampling
    L_d_chol = torch.linalg.cholesky(Sigma_d)
    
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
        # χ² distribution: sum of df squared standard normals
        chi2 = torch.sum(torch.randn(df, *shape) ** 2, dim=0)
        return normal / torch.sqrt(chi2 / df)

    # ==========================================
    # #### Train Set Generation
    # ==========================================
    for _ in range(num_sample_train):
        # 1. Initialization
        z_curr = torch.randn(d) * 0.5  # Small initial values
        
        # 2. Burn-in Phase
        for _ in range(burn_in):
            z_curr = nonlinear_transition(z_curr, coef_eps)
            
        # 3. Time Loop
        Z = torch.zeros(d, T_total)
        
        for t in range(T_total):
            z_curr = nonlinear_transition(z_curr, coef_eps)
            Z[:, t] = z_curr

        Z_list.append(Z)
        
        # Generate X: Stochastic Volatility Observation with Intercept
        # Eq: X_t = F * (exp(Z_t / 2) ⊙ 0.3 * η_t) + intercept, η_t ~ t_{df}(0, Sigma_d)
        # exp(Z/2) creates volatility scaling
        volatility = torch.exp(Z / 2)  # (d, T_total)
        eta = sample_t_distribution((d, T_total), df, L_d_chol) * obs_noise_std  # (d, T_total), scaled correlated observation noise
        # Hadamard product (element-wise)
        vol_scaled = volatility * eta  # (d, T_total)
        X = F_random @ vol_scaled + intercept.unsqueeze(1)  # (p, T_total), 加 intercept
        
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
                z_t = Z[:, t]
                mask_logits_t = W_mask @ z_t # (p,)
                mask_logits_t = mask_logits_t + b_mask # Add bias: (p,)
                mask_probs_t = torch.sigmoid(mask_logits_t) # (p,), values in [0, 1]
                mask_probs_t = mask_lower_bound + (1 - mask_lower_bound) * mask_probs_t
                mask_t = torch.bernoulli(mask_probs_t).bool() # (p,), values 0 or 1
                mask[:, t] = mask_t # Assign to mask matrix
        
        # 对于 extra_steps 部分，mask 设为全 0（全 missing），表示这些是 forecasting 目标
        if extra_steps > 0:
            mask[:, T:] = False

        mask_list.append(mask)

        X_obs = X.clone()
        X_obs[~mask] = torch.nan
        X_list.append(X_obs)
        X_complete.append(X)
        
    # ==========================================
    # ############### Test Set #################
    # ==========================================
    for _ in range(num_sample_test):
        # 1. Initialization
        z_curr = torch.randn(d) * 0.5
        
        # 2. Burn-in
        for _ in range(burn_in):
            z_curr = nonlinear_transition(z_curr, coef_eps)
            
        # 3. Time Loop
        Z = torch.zeros(d, T_total_test)
        
        for t in range(T_total_test):
            z_curr = nonlinear_transition(z_curr, coef_eps)
            Z[:, t] = z_curr
            
        Z_list_test.append(Z)
        
        # Generate X: Stochastic Volatility Observation with Intercept
        volatility = torch.exp(Z / 2)  # (d, T_total_test)
        eta = sample_t_distribution((d, T_total_test), df, L_d_chol) * obs_noise_std  # (d, T_total_test), scaled correlated observation noise
        vol_scaled = volatility * eta  # (d, T_total_test)
        X = F_random @ vol_scaled + intercept.unsqueeze(1)  # (p, T_total_test), 加 intercept
        
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
        
    # Note: phi is None for Case 3 (Nonlinear dynamics)
    # Return Sigma_d for compatibility (no Sigma_p since no additive observation noise in Case 3)
    return [X_list, X_list_test], [mask_list, mask_list_test], [Z_list, Z_list_test], [X_complete, X_complete_test], None, F_random, [psi, W_mask, b_mask, Sigma_d]
