import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import torch.optim as optim
import sys
from pathlib import Path
sys.path.append("..")
import numpy as np


# from nets.layers import AutoRegressive_DecoderLayer
############## salt (original name is sait, interestingly) ##############
######## Only one modified:  changed to upper triangular matrix without diagonal
from nets.salt_layers import RoPE1D,RoPEMultiHeadAttention,PositionalEncoding, EncoderLayer, MultiHeadAttention,RoPEEncoderLayer
from nets.salt_layers import PositionWiseFeedForward
class InitialNoiseProcessor(nn.Module):
    """
    Initial State Generator based on theoretical burn-in distribution.
    
    根据数据生成函数，z1 是经过 burn_in 步演化后的分布：
    - 初始: Z[:, 0] ~ N(0, I)
    - 演化: Z[:, t] = phi @ Z[:, t-1] + coef_eps * noise
    - z1 实际上是 Z[:, burn_in]
    
    理论分布: z1 ~ N(mu_burn, Sigma_burn)
    """
    def __init__(self, d_model, mu=None, sigma=None):
        """
        Args:
            d_model: int, latent dimension
            mu: (d,) or (d, 1) 理论均值，如果 None 则默认为 0
            sigma: (d, d) 理论协方差，如果 None 则默认为 I
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 注册 mu 和 sigma 为 buffer（不参与训练，但会随模型保存/加载）
        if mu is None:
            mu = torch.zeros(d_model)
        else:
            mu = mu.squeeze()  # 确保是 (d,) 形状
        
        if sigma is None:
            sigma = torch.eye(d_model)
        
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        
        # 预计算 Cholesky 分解用于采样
        # z = mu + L @ eps, where eps ~ N(0, I)
        # 这样 z ~ N(mu, L @ L^T) = N(mu, sigma)
        try:
            L = torch.linalg.cholesky(sigma)
        except RuntimeError:
            # 如果 sigma 不是正定的，加一个小的对角项
            L = torch.linalg.cholesky(sigma + 1e-6 * torch.eye(d_model))
        
        self.register_buffer('cholesky_L', L)

    def forward(self, batch_size, device):
        """
        采样 z1 ~ N(mu, sigma)
        
        Returns:
            z1: (batch_size, 1, d_model) 初始隐状态
        """
        # 采样标准正态
        eps = torch.randn(batch_size, self.d_model, device=device)
        
        # 转换到目标分布: z = mu + L @ eps
        # L 是 sigma 的 Cholesky 分解
        z1 = self.mu.to(device) + eps @ self.cholesky_L.T.to(device)
        
        # 调整形状 (B, 1, d)
        return z1.unsqueeze(1)
    
    def set_distribution(self, mu, sigma):
        """
        动态更新分布参数（用于根据不同数据集调整）
        
        Args:
            mu: (d,) 新的均值
            sigma: (d, d) 新的协方差
        """
        device = self.mu.device
        mu = mu.squeeze().to(device)
        sigma = sigma.to(device)
        
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)
        
        # 重新计算 Cholesky
        try:
            L = torch.linalg.cholesky(sigma)
        except RuntimeError:
            L = torch.linalg.cholesky(sigma + 1e-6 * torch.eye(self.d_model, device=device))
        
        self.cholesky_L.copy_(L)


# 保留旧版本的实现作为备选
class InitialNoiseProcessorLearnable(nn.Module):
    """
    Learnable Initial State Generator (旧版本，通过MLP学习).
    Structure: h_0 = MLP(Standard_Gaussian_Noise)
    
    🔥 改进: 添加输出 clamp 和小初始化，防止数值爆炸
    """
    def __init__(self, d_model, hidden_ratio=5):
        super().__init__()
        
        self.d_model = d_model
        hidden_dim = int(d_model * hidden_ratio)
        
        self.noise_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # 🔥 小初始化: 让初始输出接近 0，逐渐学习
        nn.init.normal_(self.noise_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.noise_mlp[-1].bias)

    def forward(self, batch_size, device):
        z_noise = torch.randn(batch_size, self.d_model, device=device)
        perturbation = self.noise_mlp(z_noise)
        h_0 = perturbation
        
        # 🔥 数值稳定性: 防止初始状态爆炸
        h_0 = torch.clamp(h_0, min=-10.0, max=10.0)
        
        return h_0.unsqueeze(1)
    
#
#### 0.8707
class SSM_Filter(nn.Module):
    def __init__(
        self,
        time_dim,
        feature_dim,
        keep_dim,
        d_model,
        # ... 其他参数保持不变 ...
        d_inner, n_head, d_k, d_v, dropout, n_groups, n_group_inner_layers, **kwargs
    ):
        super().__init__()
        # ... (前面的参数保存和 Encoder Stack 保持不变) ...
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        #####################--->>>>>>> original encoder
        # self.encoder_layer_stack = nn.ModuleList([
        #     EncoderLayer(
        #         time_dim, 2 * feature_dim, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs
        #     ) for _ in range(n_groups)
        # ])
        #####################---<<<<<<<<<<<<
        if 'diagonal_attention_mask' in kwargs:
            kwargs.pop('diagonal_attention_mask')
        self.encoder_layer_stack = nn.ModuleList([
            RoPEEncoderLayer(
                d_time=time_dim, 
                d_feature=2*feature_dim, # 注意这里可能是 embedding 后的维度 d_model，请检查你的传参逻辑
                d_model=d_model, 
                d_inner=d_inner, 
                n_head=n_head, 
                d_k=d_k, 
                d_v=d_v, 
                dropout=dropout, 
                attn_dropout=dropout,
                max_len=time_dim + 2,  # 🔥 传入 max_len
                diagonal_attention_mask=True, # 🔥 必须为 True 才能保证 Filtering
                **kwargs
                                ) for _ in range(n_groups) # 这里你可能有两层循环，按你原来的逻辑写
             ])
        
        self.encoder_embedding = nn.Linear(2 * feature_dim, d_model)
        #### no need position embedding anymore
        # self.position_enc = PositionalEncoding(self.d_model, n_position=self.time_dim)
        
        # ============================================================
        # Mask 投影层
        # ============================================================
        # 将原始 Mask (p维度) 映射到 d_model 维度，用于最后一步提醒网络
        self.mask_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU()
        )

        # ============================================================
        # 2. 输出层架构 (Trend + Mask-Aware Residual)
        # ============================================================
        self.head_trend = nn.Linear(d_model, keep_dim)
        
        # 随机残差融合层
        # 输入: Transformer Feat(d_model) + Noise(keep_dim) + Mask Feat(d_model)
        # 维度增加: 2*d_model + keep_dim
        self.noise_fusion = nn.Sequential(
            nn.Linear(2 * d_model + keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.head_residual = nn.Linear(d_model, keep_dim)
        
        # 【微调】Small-Init 稍微调大一点点 (0.02 -> 0.05)
        # 配合 Mask Shortcut，能更果断地在缺失位置输出大方差
        nn.init.normal_(self.head_residual.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.head_residual.bias, 0.0)

    def forward(self, x, mask):
        B, T, _ = x.shape
        device = x.device
        
        # --- Input Processing ---
        x_enc = x
        mask_enc = mask.float()
        x_noise = x_enc * mask_enc + (1 - mask_enc) * torch.randn_like(x_enc)
        
        if mask_enc.all():
            noise_channel = torch.randn_like(x_enc)
            masked_x = torch.cat([x_noise, noise_channel], dim=-1)
        else:
            masked_x = torch.cat([x_noise, mask_enc], dim=-1)
            
        # --- Transformer Encoding ---
        Z_feat = self.encoder_embedding(masked_x)
        # Z_feat = self.position_enc(Z_feat)
        
        for encoder_layer in self.encoder_layer_stack:
            for _ in range(self.n_group_inner_layers):
                Z_feat, _ = encoder_layer(Z_feat) 

        # ============================================================
        # 3. 输出生成 (Mask-Aware)
        # ============================================================
        # Step A: 确定性趋势
        z_trend = self.head_trend(Z_feat)
        
        # Step B: 准备 Mask 特征 (提醒网络哪里是缺失的)
        mask_feat = self.mask_proj(mask_enc) # (B, T, d_model)
        
        # Step C: 生成独立噪声
        noise = torch.randn(B, T, self.keep_dim, device=device)
        
        # Step D: 计算随机残差 (Feature + Noise + Mask)
        # 显式拼接 Mask 特征，强迫网络根据缺失情况调整噪声幅度
        fusion_input = torch.cat([Z_feat, noise, mask_feat], dim=-1)
        
        noise_feat = self.noise_fusion(fusion_input)
        z_residual = self.head_residual(noise_feat)
        
        # Step E: 叠加 (🔥 不使用 patient adapter，Z space 不是 identifiable)
        Z_final = z_trend + z_residual
        
        # 🔥 数值稳定性: 防止 Z 爆炸
        Z_final = torch.clamp(Z_final, min=-100.0, max=100.0)
        
        return Z_final
    
    
##### sanyou -- simulation
#### add rope (relative weights)

class SSM_Smoother(nn.Module):
    """
    SSM Smoother - V1 版本 (residual MLP)
    
    公式: z_t^s = z_t^f + residual_mlp(z_t^f, z_{t+1}^s, noise)
    
    这是最简单的版本，直接学习残差
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, n_head=4, d_k=16, d_v=16, dropout=0.1, transition_g=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim 
        self.time_dim = time_dim
        
        # ============================================================
        # V1 版本: Residual MLP
        # ============================================================
        self.residual_mlp = nn.Sequential(
            nn.Linear(keep_dim * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, keep_dim)
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        
        # 🔥 Cache (保留用于兼容)
        self._cache = {}
        
        # 保留 transition_g 引用 (用于兼容，但 V1 不使用)
        # self.transition_g = transition_g
        # self.own_transition = transition_g is None
        
        # ============================================================
        # 🔥 方案 E: RTS-Inspired Smoother (注释备份)
        # ============================================================
        # # 1. 共享的 transition 网络 (用于预测 z_{t+1}^p)
        # if transition_g is not None:
        #     self.transition_g = transition_g
        #     self.own_transition = False
        # else:
        #     self.transition_g = nn.Linear(keep_dim, keep_dim, bias=False)
        #     nn.init.eye_(self.transition_g.weight)
        #     self.own_transition = True
        # 
        # # 2. Gain 网络: 学习 J_t ∈ (0, 1)
        # self.gain_net = nn.Sequential(
        #     nn.Linear(keep_dim * 2, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, keep_dim),
        # )
        # nn.init.zeros_(self.gain_net[-1].weight)
        # nn.init.zeros_(self.gain_net[-1].bias)
        # ============================================================

    # ============================================================
    # V1 forward: z_s_t = z_f_t + residual
    # ============================================================
    def forward(self, z_f_t, z_s_next, noise=None, noise_scale=0.1):
        """
        V1 版本: z_t^s = z_t^f + residual_mlp(z_t^f, z_{t+1}^s, scaled_noise)
        
        Args:
            z_f_t: [B, keep_dim] - 当前时刻的 filter 输出 z_t^f
            z_s_next: [B, keep_dim] - 下一时刻的 smoothed z_{t+1}^s
            noise: [B, keep_dim] - 随机噪声 (用于生成模型)
            noise_scale: float - 噪声缩放因子
        
        Returns:
            z_s_t: [B, keep_dim] - 当前时刻的 smoothed z_t^s
        """
        B = z_f_t.shape[0]
        device = z_f_t.device
        
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        scaled_noise = noise * noise_scale
        fusion_input = torch.cat([z_f_t, z_s_next, scaled_noise], dim=-1)
        z_residual = self.residual_mlp(fusion_input)
        z_s_t = z_f_t + z_residual
        
        # 数值稳定性
        z_s_t = torch.clamp(z_s_t, min=-100.0, max=100.0)
        
        return z_s_t
    
    # ============================================================
    # 🔥 方案 E forward (RTS-Inspired) 注释备份
    # ============================================================
    # def forward_rts_inspired(self, z_f_t, z_s_next, noise=None, noise_scale=0.1):
    #     """
    #     RTS-Inspired: z_t^s = z_t^f + gain * (z_{t+1}^s - g(z_t^f))
    #     """
    #     if self.own_transition:
    #         z_pred_next = self.transition_g(z_f_t)
    #     else:
    #         with torch.no_grad():
    #             z_pred_next = self.transition_g(z_f_t, noise=None)
    #     
    #     innovation = z_s_next - z_pred_next
    #     gain_input = torch.cat([z_f_t, z_s_next], dim=-1)
    #     gain = torch.sigmoid(self.gain_net(gain_input))
    #     z_s_t = z_f_t + gain * innovation
    #     z_s_t = torch.clamp(z_s_t, min=-100.0, max=100.0)
    #     
    #     return z_s_t
    # ============================================================
    
    # ============================================================
    # 🔥 Cache 方法 (用于兼容 FBI_SSM 的 run_backward_pass)
    # ============================================================
    #         z_next: [B, keep_dim] - Z_{t+1}
    #         
    #     Returns:
    #         z_curr: [B, keep_dim] - smoothed Z_t
    #     """
    #     assert self._cache, "Must call precompute_history_features first!"
    #     
    #     B = self._cache['B']
    #     device = z_next.device
    #     
    #     # 从 cache 中取出 H[:, :t+1, :] 对应的 K, V
    #     key_emb = self._cache['key_emb_full'][:, :t+1, :]  # [B, t+1, d_model]
    #     value_emb = self._cache['value_emb_full'][:, :t+1, :]  # [B, t+1, d_model]
    #     
    #     t_len = t + 1  # history length
    #     
    #     # Query projection (这个必须每次计算，因为 z_next 不同)
    #     z_next_emb = self.query_proj(z_next).unsqueeze(1)  # [B, 1, d_model]
    #     
    #     # Cross attention
    #     context, _ = self.cross_attn(
    #         q=z_next_emb,
    #         k=key_emb,
    #         v=value_emb,
    #         attn_mask=None,
    #         alibi_bias=None,
    #         start_pos_q=t_len,
    #         start_pos_k=0
    #     )
    #     context = context.squeeze(1)  # [B, d_model]
    #     
    #     # Trend
    #     det_input = torch.cat([context, z_next], dim=-1)
    #     z_trend = self.trend_net(det_input)
    #     
    #     # Noise residual
    #     noise = torch.randn(B, self.keep_dim, device=device)
    #     fusion_input = torch.cat([context, z_next, noise], dim=-1)
    #     noise_feat = self.noise_fusion(fusion_input)
    #     z_residual = self.head_residual(noise_feat)
    #     
    #     # Combine
    #     z_curr = z_trend + z_residual
    #     z_curr = torch.clamp(z_curr, min=-100.0, max=100.0)
    #     
    #     return z_curr
    
    # def forward_old(self, H, z_next, noise=None):
    #     """
    #     旧版 forward 接口 (注释保留)
    #     
    #     H: [B, t+1, keep_dim] - History-- X_{1:t}
    #     z_next: [B, keep_dim] - Z_{t+1}
    #     """
    #     B = z_next.shape[0]
    #     device = z_next.device
    #     t_len = H.shape[1]
    #     
    #     # --- A. 准备特征 ---
    #     z_next_emb = self.query_proj(z_next).unsqueeze(1)   # (B,1, d_model)
    #     key_emb = self.key_proj(H)  # (B, t+1, d_model)
    #     value_emb = self.value_proj(H)
    #     
    #     # Cross attention with RoPE
    #     context, _ = self.cross_attn(
    #         q=z_next_emb,
    #         k=key_emb,
    #         v=value_emb,
    #         attn_mask=None,
    #         alibi_bias=None,
    #         start_pos_q=t_len,
    #         start_pos_k=0
    #     )
    #     context = context.squeeze(1)  # (B, d_model)
    #     
    #     # --- B. 计算确定性趋势 ---
    #     det_input = torch.cat([context, z_next], dim=-1)
    #     z_trend = self.trend_net(det_input)
    #     
    #     # --- C. 计算随机残差 ---
    #     if noise is None:
    #         noise = torch.randn(B, self.keep_dim, device=device)
    #         
    #     fusion_input = torch.cat([context, z_next, noise], dim=-1)
    #     noise_feat = self.noise_fusion(fusion_input)
    #     z_residual = self.head_residual(noise_feat)
    #     
    #     # --- D. Combine ---
    #     z_curr = z_trend + z_residual
    #     z_curr = torch.clamp(z_curr, min=-100.0, max=100.0)
    #     
    #     return z_curr

    def precompute_history_features(self, H_full, time_stamps=None):
        """
        🔥 新版不需要 precompute，但保留接口兼容性
        只需存储 H_full 供 backward loop 使用
        """
        self._cache = {'H_full': H_full}
    
    def clear_cache(self):
        """清理 cache"""
        self._cache = {}


##################################################Decoder


# ============================================================
# 🔥 ResidualMLP: 真正的残差 MLP，防止梯度消失
# ============================================================
class ResidualBlock(nn.Module):
    """单个残差块: x + MLP(x)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm residual connection
        return x + self.dropout(self.mlp(self.norm(x)))


class ResidualMLP(nn.Module):
    """
    深层残差 MLP (Deep Residual MLP)
    
    结构: Input -> Proj_in -> [ResBlock] * n_layers -> Proj_out
    
    相比 nn.Sequential 堆叠 Linear 的优势:
    1. 残差连接防止梯度消失
    2. LayerNorm 稳定训练
    3. 任意加深而不会训练崩溃
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.proj_in(x)
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.proj_out(x)
        return x


#########################>>>>>>>>>>>>>>>>>> try to 打造的一个全能版本
#########################>>>>>>>>>>>>>>>>>> try to 打造的一个全能版本
#########################>>>>>>>>>>>>>>>>>> try to 打造的一个全能版本
#########################>>>>>>>>>>>>>>>>>> try to 打造的一个全能版本
# ============================================================
# #### [新增] 必须定义 ResidualMLP，否则你的代码会报错
# ============================================================
# class ResidualMLP(nn.Module):
#     """
#     ResMLPBlock: y = x + MLP(Norm(x))
#     用于构建深层生成网络，防止梯度消失
#     """
#     def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3, dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Linear(in_dim, hidden_dim)
        
#         self.blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(hidden_dim),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.Dropout(dropout)
#             ) for _ in range(n_layers)
#         ])
        
#         self.output_proj = nn.Linear(hidden_dim, out_dim)
        
#     def forward(self, x):
#         h = self.input_proj(x)
#         for block in self.blocks:
#             h = h + block(h) # Residual Connection
#         return self.output_proj(h)
##### update january-9th-2026
class SSM_Decoder_U(nn.Module):
    def __init__(self, feature_dim, z_dim, d_model=64, n_heads=4, dropout=0, hard_replace_u=False,
                 modality_dropout=0.0, force_alpha=None):
        """
        通用隐式插补器 (Universal Implicit Imputer)
        - 能够拟合任意分布 (Multimodal, Skewed)
        - 能够处理复杂非线性关系
        - 能够处理非加性噪声 (Noise 深度融合)
        - 鲁棒处理稀疏观测 (Z-Gating)
        - hard_replace_u=True: 硬替换 (观测值保留，只生成缺失值)
        - hard_replace_u=False: 纯生成式输出 (无硬替换) [默认]
        - modality_dropout: 训练时对 x_base_spatial 的 dropout 比例 (默认 0.3)
        - force_alpha: 强制设置 alpha 值 (用于调试/极端实验)
            - None: 正常使用 gating network
            - 1.0: x_base 完全来自 dynamic (Z)
            - 0.0: x_base 完全来自 spatial (X_obs)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.hard_replace_u = hard_replace_u  # 保存参数
        self.modality_dropout = modality_dropout  # 新增
        self.force_alpha = force_alpha  # 新增
        
        # [Fix Bug] 噪声维度提升
        # 8维足以拟合任意复杂的双峰/多峰分布
        self.noise_dim = 32
        # ============================================================
        # 🔥 [NEW] Patient-Level Mean & Std Estimator
        # ============================================================
        # 从观测值估计病人特有的 per-feature mean 和 std
        # 解决问题：某病人 creatinine 一直是 -0.3，但全局 mean=0
        # Mean Estimator: 让预测中心偏向病人特有的均值
        # Std Estimator: 让 CI 宽度适应病人特有的变异程度
        
        # 输入特征：[obs_mean, obs_std, obs_ratio] per feature
        self.patient_mean_estimator = nn.Sequential(
            nn.Linear(feature_dim * 3, d_model),  # 输入: [obs_mean, obs_std, obs_ratio]
            nn.GELU(),
            nn.Linear(d_model, feature_dim),      # 输出: per-feature mean offset
        )
        
        self.patient_std_estimator = nn.Sequential(
            nn.Linear(feature_dim * 3, d_model),  # 输入: [obs_mean, obs_std, obs_ratio]
            nn.GELU(),
            nn.Linear(d_model, feature_dim),      # 输出: per-feature std scale (log-space)
        )
        
        # ============================================================
        # 1. Dual-Stream Base
        # ============================================================
        
        # A. Spatial Stream (主力): X_obs -> Base
        self.spatial_projector = nn.Sequential(
            nn.Linear(feature_dim * 2, d_model), nn.GELU(),
            nn.Linear(d_model, feature_dim)
        )
        
        # B. Dynamic Stream (救兵): Z -> Base
        self.z_projector = nn.Sequential(
            nn.Linear(z_dim, d_model), nn.GELU(),
            nn.Linear(d_model, feature_dim)
        )
        
        # C. Adaptive Gating (智能判官)
        # 决定信 X_t^o 还是信 Z
        self.gating_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            # nn.Linear(d_model, feature_dim), 
            nn.Linear(d_model, 1), 
            nn.Sigmoid()
        )

        # ============================================================
        # 2. Rich Embeddings (融合你代码的优点)
        # ============================================================
        self.res_proj = nn.Linear(1, d_model)              # 残差投影
        self.position_embedding = nn.Parameter(torch.randn(1, feature_dim, d_model)) # 你的特征身份证
        self.mask_embedding = nn.Embedding(2, d_model)     # 状态身份证
        self.z_ctx_proj = nn.Linear(z_dim, d_model)        # Z 上下文投影

        # [修改 2] 新增特征级缩放参数
        # 允许每个特征有自己独立的 Scale 和 Bias，帮助它逃离 0 点
        # self.feature_scale = nn.Parameter(torch.ones(1, 1, feature_dim)) # <<<<< 新增
        self.feature_bias = nn.Parameter(torch.zeros(1, 1, feature_dim))
        # 初始化为 1.0，允许网络自动学习每个特征需要多少随机性
        self.noise_scale = nn.Parameter(torch.ones(1, feature_dim, 1))  # [1, p, 1]
        
        # ============================================================
        # 3. Interaction (Attention)
        # ============================================================
        # 让特征之间互相"商量"怎么填空
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # ============================================================
        # 4. Implicit Generator (核心黑科技)
        # ============================================================
        # 这不再是简单的投影，而是一个条件生成网络
        # 它学习映射: f(Context, Noise) -> X_target
        
        # 噪声投影
        self.noise_proj = nn.Linear(self.noise_dim, d_model)
        
        # 🔥 深层残差 MLP (Deep Residual MLP)
        # 真正的残差连接，防止梯度消失
        # 只有足够深，才能模拟复杂的非高斯分布和非加性噪声
        self.generator = ResidualMLP(
            in_dim=d_model * 2,      # Context + Noise
            hidden_dim=d_model * 2,  # 隐藏层维度
            out_dim=1,               # 输出修正量
            n_layers=3,              # 3 个残差块 (相当于 6 层 Linear)
            dropout=dropout
        )
        ############# the optimization fix #############
        self._init_weights()
    def _init_weights(self):
        """
        严谨的初始化策略：
        1. 让 Gating Net 在初始状态下偏向 Spatial (Z 通道权重低)
        2. 让 Patient-Level Estimator 初始输出为 0 (不改变预测)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # ============================================================
        # 🔥 [改进 1] Patient-Level Mean Estimator: 初始化最后一层为 0
        # ============================================================
        # patient_mean_estimator[-1] 是最后一个 Linear 层
        # 初始时输出 0，意味着不添加任何 mean offset
        # 这让模型可以安全地从"无偏移"状态开始学习
        nn.init.zeros_(self.patient_mean_estimator[-1].weight)
        nn.init.zeros_(self.patient_mean_estimator[-1].bias)
        print("✅ Patient Mean Estimator: Last layer initialized to zeros (start from no offset)")
        
        # ============================================================
        # 🔥 [改进 2] Patient-Level Std Estimator: 初始化最后一层为 0
        # ============================================================
        # 输出 0 经过 exp() 后变成 1.0，即初始不缩放
        # 这让模型可以安全地从"标准尺度"状态开始学习
        nn.init.zeros_(self.patient_std_estimator[-1].weight)
        nn.init.zeros_(self.patient_std_estimator[-1].bias)
        print("✅ Patient Std Estimator: Last layer initialized to zeros (exp(0)=1.0, no scaling)")
        
        # ============================================================
        # 🔥 [改进 3] Gating Network: 初始化偏向 Spatial Stream
        # ============================================================
        # gating_net 结构: [Linear, GELU, Linear(输出层), Sigmoid]
        # 我们要修改的是索引为 2 的那个 Linear 的 bias
        output_layer = self.gating_net[2] 
        
        # Sigmoid(-0.5) ≈ 0.38，意味着初始 alpha ≈ 0.38
        # 初始时 Spatial 权重 62%，Z 权重 38%
        # 比 -5.0 更温和，允许两个 stream 都有贡献
        nn.init.constant_(output_layer.bias, -0.5)
        print("✅ Gating Network: Initialized with Bias -0.5 (Spatial stream favored, alpha≈0.38)")
        
    def forward(self, z_t,x_obs, mask_obs, noise=None):
        """
        Input:
            x_obs: [B, T, p]
            mask_obs: [B, T, p]
            z_t: [B, T, z_dim]
        """
        B, T, p = x_obs.shape
        
        # ============================================================
        # 🔥 [NEW] Step 0: Compute Patient-Level Mean & Std
        # ============================================================
        # 从观测值估计这个病人的 per-feature mean 和 std
        
        # 1. 计算观测均值
        obs_sum = (x_obs * mask_obs).sum(dim=1)  # [B, p] - 每个特征的观测值总和
        obs_count = mask_obs.sum(dim=1).clamp(min=1)  # [B, p] - 每个特征的观测数量
        patient_obs_mean = obs_sum / obs_count  # [B, p] - 每个特征的观测均值
        obs_ratio = obs_count / T  # [B, p] - 观测率 (0~1)
        
        # 2. 计算观测标准差
        # std = sqrt(E[X^2] - E[X]^2)
        obs_sq_sum = ((x_obs ** 2) * mask_obs).sum(dim=1)  # [B, p]
        patient_obs_var = (obs_sq_sum / obs_count) - (patient_obs_mean ** 2)  # [B, p]
        patient_obs_std = torch.sqrt(patient_obs_var.clamp(min=1e-6))  # [B, p] - 防止 sqrt(负数)
        
        # 3. 用 MLP 学习如何利用这些统计量
        patient_stats_input = torch.cat([patient_obs_mean, patient_obs_std, obs_ratio], dim=-1)  # [B, 3p]
        
        # Mean offset: 让预测中心偏向病人特有的均值
        mean_correction = self.patient_mean_estimator(patient_stats_input)  # [B, p]
        # patient_obs_mean: [B, p] -> unsqueeze(1) -> [B, 1, p] -> expand -> [B, T, p]
        patient_mean_offset = patient_obs_mean.unsqueeze(1).expand(B, T, p) + mean_correction.unsqueeze(1).expand(B, T, p)  # [B, T, p]
        
        # Std scale: 让噪声/CI 宽度适应病人特有的变异程度
        # 使用 exp() 确保 scale > 0，输出在 log-space
        # 🔥 [改进 4] 添加 clamp 防止 exp 爆炸 (当 MLP 输出过大时)
        patient_std_scale_raw = self.patient_std_estimator(patient_stats_input)  # [B, p]
        patient_std_scale = torch.exp(patient_std_scale_raw.clamp(-5, 5))  # clamp 保护
        patient_std_scale = patient_std_scale.unsqueeze(1).expand(B, T, p)  # [B, T, p]
        
        # Flatten [N, ...]
        x_obs = x_obs * mask_obs  # 确保缺失值为0
        x_flat = x_obs.reshape(B*T, p)
        m_flat = mask_obs.reshape(B*T, p)
        z_flat = z_t.reshape(B*T, -1)
        
        # ============================================================
        # Step 1: Robust Base Prediction (Anchor)
        # ============================================================
        
        # 1. Spatial Guess
        spatial_in = torch.cat([x_flat, m_flat.float()], dim=-1)
        x_base_spatial = self.spatial_projector(spatial_in)
        
        # 2. Dynamic Guess
        x_base_dynamic = self.z_projector(z_flat)
        
        # ============================================================
        # 🔥【插入在这里】 Modality Dropout
        # ============================================================
        # 必须在算出 x_base_spatial 之后，但在计算 res_spatial 之前
        if self.training and self.modality_dropout > 0:
            # 生成一个 mask: [B*T, 1]
            # torch.rand > dropout_rate 意味着 (1-dropout_rate) 的概率保留
            dropout_mask = (torch.rand(B*T, 1, device=x_obs.device) > self.modality_dropout).float()
            
            # 如果 mask 是 0，x_base_spatial 就变成了 0
            x_base_spatial = x_base_spatial * dropout_mask
        
        ############### gating
        # 3. Gating Logic
        # 计算 Spatial 的残差，评估其可靠性
        res_spatial = (x_flat - x_base_spatial) * m_flat.float()
        
        # 准备 Gating 输入
        emb_mask = self.mask_embedding(m_flat.long())
        emb_res_spatial = self.res_proj(res_spatial.unsqueeze(-1))
        
        # 计算权重: mask=0 或 spatial误差大 -> alpha变大 -> 引入 Z
        gate_in = torch.cat([emb_mask, emb_res_spatial], dim=-1)
        alpha = self.gating_net(gate_in)
        #### 去掉最后一维，变成 [N, p]
        alpha = alpha.squeeze(-1) # <--- 修改这里！
        
        # 🔥 支持强制设置 alpha (用于极端实验)
        if self.force_alpha is not None:
            alpha = torch.full_like(alpha, self.force_alpha)
        
        # 融合 Base
        # print("shape of x_base_spatial:", x_base_spatial.shape)
        # print("shape of x_base_dynamic:", x_base_dynamic.shape)
        # print("shape of alpha:", alpha.shape)
        # x_base = x_base_spatial + alpha * x_base_dynamic
        x_base = (1 - alpha) * x_base_spatial + alpha * x_base_dynamic
        
        # ============================================================
        # Step 2: Context Construction (借用你的 Embedding 设计)
        # ============================================================
        # 计算相对于 Base 的残差
        residual = (x_flat - x_base) * m_flat.float()
        
        # 构建丰富的上下文
        emb_res = self.res_proj(residual.unsqueeze(-1)) # 残差信息
        # emb_mask (已计算)                             # 状态信息
        emb_pos = self.position_embedding.expand(B*T, -1, -1) # 特征身份 (Identity)
        emb_z = self.z_ctx_proj(z_flat).unsqueeze(1)    # 全局动力学上下文
        
        # 融合: Context = Res + Mask + Identity + Z
        # [N, p, d_model]
        context = emb_res + emb_mask + emb_pos + emb_z
        
        # ============================================================
        # Step 3: Interaction (Attention)
        # ============================================================
        # 修正上下文，捕捉特征间依赖
        context_h, _ = self.feature_attention(context, context, context)
        
        # ============================================================
        # Step 4: Implicit Generation (见证奇迹的时刻)
        # ============================================================
        
        # 1. 注入噪声
        if noise is None:
            # [B*T, p, noise_dim]
            noise = torch.randn(B*T, p, self.noise_dim, device=x_obs.device)
        else:
            if noise.dim() == 2: noise = noise.unsqueeze(-1)
        
        # 🔥 [防止 Mean Collapse] 应用可学习的特征级噪声缩放
        # noise_scale: [1, p, 1], 广播到 [B*T, p, noise_dim]
        # 让每个特征学习自己需要多少随机性
        # 🔥 [改进 5] 使用 softplus 确保 noise_scale > 0
        noise = noise * F.softplus(self.noise_scale)  # softplus 保证正数，初始值约 1.31
            
        # 2. 将噪声映射到高维
        noise_emb = self.noise_proj(noise) # [N, p, d_model]
        
        # 3. 拼接 (Context + Noise)
        # 这一步实现了"非加性噪声"。噪声不再是加在结果上，而是作为原料输入进去。
        # [N, p, 2*d_model]
        gen_input = torch.cat([context_h, noise_emb], dim=-1)
        
        # 4. 非线性映射 (The Sculptor)
        # MLP 根据 Context 的指示 (比如 mask=1)，决定如何处理 Noise。
        # 如果 mask=1，MLP 可以学会输出恒定的 delta，忽略 Noise (实现 Soft Constraint)
        # 如果 mask=0，MLP 可以把 Noise 扭曲成双峰、偏态等任意分布。
        x_delta = self.generator(gen_input).squeeze(-1)
        x_delta = x_delta.reshape(B, T, p)
        # 3. 【关键修正】应用 feature_bias
        #### update 
        x_delta = x_delta * patient_std_scale  # 1. 先缩放波动
        
        # ============================================================
        # Step 5: Final Output (No Hard Replacement!)
        # ============================================================
        # 完全由网络生成的预测值
        # 包含了 Base 的趋势 + Delta 的精修(及随机性)
        x_base = x_base.reshape(B, T, p)
        
        # 🔥 [NEW] 加上 Patient-Level Mean Offset
        # key changes: 
        x_gen = x_base + x_delta + patient_mean_offset +  self.feature_bias 
        
        # 🔥 数值稳定性: 防止输出爆炸
        x_gen = torch.clamp(x_gen, min=-100.0, max=100.0)
        
        # return x_gen.reshape(B, T, p)
        x_gen = x_gen.reshape(B, T, p)
        if self.hard_replace_u:
            return x_obs * mask_obs.float() + x_gen * (1 - mask_obs.float())
        else:
            return x_gen
########################################<<<<<<<<<<<<<< 全能版本
########################################<<<<<<<<<<<<<< 全能版本
########################################<<<<<<<<<<<<<< 全能版本
########################################<<<<<<<<<<<<<< 全能版本
########################################<<<<<<<<<<<<<< 全能版本
################################################## decoder################################################## decoder################################################## decoder
################################################## decoder################################################## decoder
################################################## decoder
################################################## decoder
################################################## decoder
################################################## decoder


#######============================================================
####### 🔥 统一的自适应观测网络 (Adaptive Observation Network)
#######============================================================
class SSM_Out_Fx_Adaptive(nn.Module):
    """
    统一的自适应观测解码器 - 能自动适应线性和非线性观测模型。
    
    核心思想：使用 Gating Network 自动平衡线性和非线性路径
    - 线性路径 (Linear Shortcut): X = W @ Z，适合线性观测模型
    - 非线性路径 (MLP): X = MLP(Z)，适合非线性观测模型
    - Gating: α = σ(gate_param)，决定两条路径的权重
    
    优点：
    1. 对线性数据：gate 关闭 MLP → 退化为线性模型，不会过拟合
    2. 对非线性数据：gate 打开 MLP → 能拟合复杂映射
    3. 自动学习 α，无需手动指定 fx_type
    """
    def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        self.d_model = d_model
        
        # ============================================================
        # 1. Linear Path (高速公路)
        # ============================================================
        # 对应经典线性观测模型: X = F @ Z + noise
        self.linear_path = nn.Linear(keep_dim, feature_dim, bias=True)
        nn.init.orthogonal_(self.linear_path.weight)
        nn.init.zeros_(self.linear_path.bias)
        
        # ============================================================
        # 2. Nonlinear Path (MLP with Residual)
        # ============================================================
        hidden_dim = max(d_model * 2, 64)
        
        # 输入投影
        self.mlp_input = nn.Sequential(
            nn.LayerNorm(keep_dim),
            nn.Linear(keep_dim, hidden_dim),
            nn.GELU()
        )
        
        # Residual MLP 块 (深度适中)
        self.mlp_res1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.mlp_res2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 输出投影
        self.mlp_output = nn.Linear(hidden_dim, feature_dim)
        
        # 小初始化 MLP 输出，让网络起始更偏向线性
        nn.init.normal_(self.mlp_output.weight, std=0.02)
        nn.init.zeros_(self.mlp_output.bias)
        
        # ============================================================
        # 3. Gating Network (自适应门控)
        # ============================================================
        # gate = σ(gate_param) ∈ [0, 1]
        # output = (1 - gate) * linear + gate * mlp
        # 
        # 初始化为负数，让 gate 初始接近 0（偏向线性）
        # 训练时如果需要非线性，gate 会自动增大
        self.gate_param = nn.Parameter(torch.full((feature_dim,), -3.0))
        
        # ============================================================
        # 4. Noise Modeling (观测噪声)
        # ============================================================
        # Log-space 参数，初始化为小值
        self.log_noise_scale = nn.Parameter(torch.full((feature_dim,), -3.0))
        
        # Context-dependent noise modulation
        self.noise_modulator = nn.Sequential(
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim)
        )
        # 小初始化
        nn.init.normal_(self.noise_modulator[-1].weight, std=0.01)
        nn.init.zeros_(self.noise_modulator[-1].bias)
        
    def forward(self, z, noise=None):
        """
        Args:
            z: [B, T, keep_dim] 隐状态
            noise: [B, T, feature_dim] 可选的外部噪声
        Returns:
            x_gen: [B, T, feature_dim] 生成的观测
        """
        # ============================================================
        # Step 1: Linear Path
        # ============================================================
        x_linear = self.linear_path(z)  # [B, T, feature_dim]
        
        # ============================================================
        # Step 2: Nonlinear Path
        # ============================================================
        h = self.mlp_input(z)           # [B, T, hidden_dim]
        h = h + self.mlp_res1(h)        # Residual
        h = h + self.mlp_res2(h)        # Residual
        x_mlp = self.mlp_output(h)      # [B, T, feature_dim]
        
        # ============================================================
        # Step 3: Gated Combination
        # ============================================================
        # gate ∈ [0, 1]，控制线性和非线性的混合比例
        gate = torch.sigmoid(self.gate_param)  # [feature_dim]
        
        # 混合：当 gate ≈ 0 时退化为线性，gate ≈ 1 时为纯 MLP
        x_mean = (1 - gate) * x_linear + gate * x_mlp
        
        # ============================================================
        # Step 4: Noise
        # ============================================================
        if noise is None:
            noise = torch.randn_like(x_mean)
        
        # Base scale
        base_scale = torch.exp(self.log_noise_scale)  # [feature_dim]
        
        # Context-dependent modulation (小扰动)
        scale_delta = 0.1 * torch.tanh(self.noise_modulator(z))
        noise_scale = base_scale * (1 + scale_delta)
        
        x_gen = x_mean + noise_scale * noise
        
        return x_gen
    
    def get_gate_info(self):
        """返回 gate 状态，用于诊断和可视化"""
        gate = torch.sigmoid(self.gate_param).detach()
        return {
            'gate_mean': gate.mean().item(),
            'gate_min': gate.min().item(),
            'gate_max': gate.max().item(),
            'gate_values': gate.cpu().numpy()
        }






class ResMLPBlock(nn.Module):
    """简洁的 Residual Block: Norm -> Linear -> GELU -> Dropout -> Linear -> Residual"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class SSM_Out_Fx_MLP(nn.Module):
    def __init__(self, feature_dim, keep_dim, d_model):
        super().__init__()
        
        # ======================================================
        # 1. Pattern Generator (负责生成波动的形状)
        # ======================================================
        # 输入: Z (上下文) + Noise (随机性)
        # 输出: 归一化的波形 (Normalized Shape)
        self.noise_dim = feature_dim
        self.input_proj = nn.Linear(keep_dim + self.noise_dim, d_model)
        
        self.backbone = nn.Sequential(
            ResMLPBlock(d_model),
            ResMLPBlock(d_model),
            ResMLPBlock(d_model)
        )
        self.head_pattern = nn.Linear(d_model, feature_dim)
        
        # ======================================================
        # 2. Patient Adapter (负责生成病人的专属基准)
        # ======================================================
        # 输入: 仅 Z (不需要噪声，因为基准是确定的事实)
        # 输出: 该时刻该病人的 Scale(方差) 和 Bias(均值)
        # 这是一个轻量级的 Side Network
        self.patient_adapter = nn.Sequential(
            nn.Linear(keep_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, feature_dim * 2) # 输出 [scale, bias]
        )
        
        # 全局兜底参数 (可选，为了让训练初期稳定)
        # self.global_scale = nn.Parameter(torch.ones(1, 1, feature_dim))
        # self.global_bias = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, z_next, noise=None):
        """
        z_next: [B, T, keep_dim] (SSM 状态，包含了病人的历史记忆)
        """
        B, T, _ = z_next.shape
        
        # ---------------------------------------------------
        # A. 计算 Patient-Specific Statistics (DC 分量)
        # ---------------------------------------------------
        # 网络会自动学会：如果 z 里暗示是病人A，则 bias 输出 1.0
        # 如果 z 里暗示是病人B，则 bias 输出 -0.3
        adapter_out = self.patient_adapter(z_next) # [B, T, 2*F]
        
        # 拆分 scale 和 bias
        p_scale, p_bias = adapter_out.chunk(2, dim=-1)
        
        # 处理 scale: 必须 > 0，且建议以 1.0 为中心
        # Softplus 保证正数，+ 0.5 让初始状态接近恒等变换
        # 🔥 数值稳定性: 添加上限 clamp
        # Training data is z-normalized (std≈1), so scale > 3 is rarely justified.
        p_scale = torch.clamp(F.softplus(p_scale) + 0.5, max=3.0)   # was 5.0
        p_bias = torch.clamp(p_bias, min=-5.0, max=5.0)              # was ±10
        
        # ---------------------------------------------------
        # B. 生成 Dynamic Pattern (AC 分量)
        # ---------------------------------------------------
        if noise is None:
            noise = torch.randn(B, T, self.noise_dim, device=z_next.device)
            
        # Early Fusion: 让 Pattern 充分利用噪声
        combined = torch.cat([z_next, noise], dim=-1)
        h = self.backbone(self.input_proj(combined))
        
        # 得到归一化的波动形状 (Centered around 0)
        raw_pattern = self.head_pattern(h)
        # Clamp backbone output before amplification to prevent extreme values.
        # For z-normalized data (std≈1), raw_pattern beyond ±3 is already > 3σ.
        raw_pattern = torch.clamp(raw_pattern, -3.0, 3.0)
        
        # ---------------------------------------------------
        # C. 融合: Pattern * Scale + Bias
        # ---------------------------------------------------
        # 1. 先应用病人的动态特征 (Instance-Level)
        #    病人A: pattern * 1.0 + 1.0
        #    病人B: pattern * 1.0 - 0.3
        x_gen = raw_pattern * p_scale + p_bias
        
        # 🔥 数值稳定性: 防止输出爆炸
        # Training data range: approximately [-26, 35] after z-normalization.
        # Clamp at ±20 covers most realistic values while preventing extreme generation.
        x_gen = torch.clamp(x_gen, min=-20.0, max=20.0)  # was ±100
        
        return x_gen

##############---------------------- clear version, maybe we can used it for simulation later
# class SSM_Out_Fx_MLP(nn.Module):
#     def __init__(self, feature_dim, keep_dim, d_model):
#         super().__init__()
#         # 1. 这里的输入维度变大了，因为我们要把 noise 拼进去
#         # 假设 noise 的维度和 feature_dim 一样，或者你可以自定义 noise_dim
#         self.noise_dim = feature_dim 
#         self.input_proj = nn.Linear(keep_dim + self.noise_dim, d_model)
        
#         # 2. Backbone (nonlinear interaction between z and noise)
#         self.backbone = nn.Sequential(
#             ResMLPBlock(d_model),
#             ResMLPBlock(d_model),
#             ResMLPBlock(d_model)
#         )
        
#         # 3. 输出层 (不再需要 mu 和 sigma 头了，直接输出 x)
#         self.head = nn.Linear(d_model, feature_dim)
        
#         # 4. 依然保留最后的可学习 Scale/Bias (物理映射)
#         self.feature_scale = nn.Parameter(torch.ones(1, 1, feature_dim))
#         self.feature_bias = nn.Parameter(torch.zeros(1, 1, feature_dim))

#     def forward(self, z_next, noise=None):
#         # z_next: [B, T, keep_dim]
#         B, T, _ = z_next.shape
        
#         # 1. 生成噪声 (如果在训练外部没传，就内部生成)
#         if noise is None:
#             noise = torch.randn(B, T, self.noise_dim, device=z_next.device)
            
#         # 2. 【关键一步】早期拼接 (Early Concatenation)
#         # 现在的输入包含了确定性信息(z)和随机性信息(noise)
#         combined_input = torch.cat([z_next, noise], dim=-1) # [B, T, keep_dim + noise_dim]
        
#         # 3. 通过网络
#         # Noise 在这里经过了非线性变换，不再是简单的加减乘除
#         h = self.input_proj(combined_input)
#         h = self.backbone(h)
        
#         # 4. 直接输出
#         x_gen = self.head(h)
        
#         # 5. 物理空间映射
#         x_gen = x_gen * self.feature_scale + self.feature_bias
        
#         return x_gen
#######----New MLP-only version for nonlinear observation models (Case 2)
# class SSM_Out_Fx_MLP(nn.Module):
#     """
#     MLP-only observation decoder for nonlinear observation models.
    
#     设计理念：
#     1. 去掉 linear_shortcut, pure MLP to lear the nonlinear output functions (如 X = F @ Z^2)
#     2. 使用 Residual MLP 块，更深更宽，能学习复杂非线性
#     3. Noise 处理:context-dependent scale + 加性噪声
#        - mean_net: 预测 E[X|Z]
#        - scale_net: 预测 Std[X|Z] only depends on Z, more general
#     4. 初始化策略：让输出尺度和输入尺度匹配
#     5. Feature-Level Scale & Bias: 每个特征独立学习自己的缩放和偏置
#     6. Patient-Level Adaptation (optional): 可以从观测数据中提取patient-level信息
#     """
#     def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0, use_patient_level=False):
#         super().__init__()
        
#         self.feature_dim = feature_dim
#         self.keep_dim = keep_dim
#         self.use_patient_level = use_patient_level
        
#         # 更宽的隐藏层
#         hidden_dim = max(d_model * 4, 128)
        
#         # ============================================================
#         # 🔥 NEW: Feature-Level Learnable Scale & Bias
#         # ============================================================
#         # 让每个特征有独立的缩放和偏置，帮助其逃离0点附近
#         # 初始化: scale=1.0, bias=0.0 (identity mapping)
#         self.feature_scale = nn.Parameter(torch.ones(1, 1, feature_dim))
#         self.feature_bias = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
#         # ============================================================
#         # 🔥 NEW (Optional): Patient-Level Statistics Extractor
#         # ============================================================
#         # 如果启用，从Z中提取patient-level的统计信息
#         # 用于调整每个样本的均值偏移和方差缩放
#         if self.use_patient_level:
#             # 从整个序列的Z中提取patient-level特征
#             # 输入: [B, T, keep_dim] -> 输出: [B, feature_dim]
#             self.patient_mean_net = nn.Sequential(
#                 nn.Linear(keep_dim, d_model),
#                 nn.GELU(),
#                 nn.Linear(d_model, feature_dim)
#             )
            
#             self.patient_std_net = nn.Sequential(
#                 nn.Linear(keep_dim, d_model),
#                 nn.GELU(),
#                 nn.Linear(d_model, feature_dim)
#             )
        
#         # ============================================================
#         # 1. Mean Network: Z → E[X|Z]
#         # ============================================================
#         # 使用 Residual 块，更好的梯度流
#         self.input_proj = nn.Linear(keep_dim, hidden_dim)
        
#         # Residual MLP blocks (更深)
#         self.res_blocks = nn.ModuleList([
#             self._make_res_block(hidden_dim, dropout) for _ in range(4)
#         ])
        
#         # 输出层
#         self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
#         # 输出尺度缩放（可学习）- 解决初始化尺度问题
#         self.output_scale = nn.Parameter(torch.ones(feature_dim))
#         self.output_bias = nn.Parameter(torch.zeros(feature_dim))
        
#         # ============================================================
#         # 2. Scale Network: Z → Std[X|Z] (context-dependent noise)
#         # ============================================================
#         # 让 noise scale 依赖于 Z，更 general
#         self.scale_net = nn.Sequential(
#             nn.Linear(keep_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 4, feature_dim)
#         )
        
#         # Scale 的基线（log space，初始化为较小值）
#         self.log_scale_base = nn.Parameter(torch.full((feature_dim,), -2.0))
        
#         # ============================================================
#         # 🔥 NEW: Per-Feature Noise Scale
#         # ============================================================
#         # 让每个特征学习自己需要多少随机性
#         # BUN, Creatinine等高变异特征可以学到更大的noise_scale
#         self.noise_scale_per_feature = nn.Parameter(torch.ones(1, feature_dim, 1))
        
#         # ============================================================
#         # 3. 初始化
#         # ============================================================
#         self._init_weights()
    
#     def _make_res_block(self, hidden_dim, dropout):
#         """创建一个 Residual MLP 块"""
#         return nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Dropout(dropout)
#         )
    
#     def _init_weights(self):
#         """初始化权重，确保输出尺度合理"""
#         # 输入投影
#         nn.init.xavier_uniform_(self.input_proj.weight)
#         nn.init.zeros_(self.input_proj.bias)
        
#         # Residual blocks - 最后一层小初始化，保持 residual 性质
#         for block in self.res_blocks:
#             for module in block:
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
#             # 最后一层小初始化
#             nn.init.normal_(block[-2].weight, std=0.02)
        
#         # 输出投影 - 稍大的初始化，让输出有足够的尺度
#         nn.init.xavier_uniform_(self.output_proj.weight, gain=2.0)
#         nn.init.zeros_(self.output_proj.bias)
        
#         # Scale network - 小初始化
#         for module in self.scale_net:
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight, gain=0.1)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
        
#         # 🔥 Patient-level networks - 小初始化
#         if self.use_patient_level:
#             for module in self.patient_mean_net:
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight, gain=0.1)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
            
#             for module in self.patient_std_net:
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight, gain=0.1)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
    
#     def forward(self, z_next, noise=None):
#         """
#         Args:
#             z_next: [B, T, keep_dim] 隐状态
#                    🔥 因果性保证：z_next 本身就包含了过去信息的压缩表示
#                    z_next 的每个元素都是 encoder_h(X_{1:t}^o, mask_{1:t}) 的输出
#             noise: [B, T, feature_dim] 可选的外部噪声
#         Returns:
#             x_gen: [B, T, feature_dim] 生成的观测
#         """
#         B, T, _ = z_next.shape
        
#         # ============================================================
#         # 🔥 Step 0 - Patient-Level Adaptation (Optional)
#         # 因果性保证：使用 cumulative mean，第 t 步只用 Z_{1:t} 的信息
#         # ============================================================
#         patient_mean_offset = 0
#         patient_std_scale = 1.0
        
#         if self.use_patient_level:
#             # Cumulative mean: 每个时间步 t 只用过去 + 当前的平均
#             # cumsum[t] = sum(z_next[:, :t+1])
#             # patient_context[t] = cumsum[t] / (t+1)
#             cumsum = torch.cumsum(z_next, dim=1)  # [B, T, keep_dim]
#             counts = torch.arange(1, T + 1, device=z_next.device, dtype=z_next.dtype).view(1, T, 1)
#             patient_context_cum = cumsum / counts  # [B, T, keep_dim]
            
#             # 预测patient-level的均值偏移和标准差缩放
#             patient_mean_offset = self.patient_mean_net(patient_context_cum)  # [B, T, feature_dim]
            
#             # Std scale in log-space, then exp
#             patient_std_scale = torch.exp(self.patient_std_net(patient_context_cum))  # [B, T, feature_dim]
        
#         # ============================================================
#         # Step 1: Mean prediction
#         # ============================================================
#         h = self.input_proj(z_next)  # [B, T, hidden_dim]
        
#         # Residual blocks
#         for block in self.res_blocks:
#             h = h + block(h)  # Residual connection
        
#         # 输出
#         x_mean = self.output_proj(h)  # [B, T, feature_dim]
        
#         # 应用可学习的尺度和偏置
#         x_mean = x_mean * self.output_scale + self.output_bias
        
#         # ============================================================
#         # Step 2: Scale prediction (context-dependent)
#         # ============================================================
#         # Scale 依赖于 Z
#         scale_delta = self.scale_net(z_next)  # [B, T, feature_dim]
        
#         # 总 scale = exp(log_base + delta)，确保正数
#         # 使用 softplus 更稳定
#         log_scale = self.log_scale_base + 0.1 * scale_delta  # delta 贡献较小
#         x_scale = torch.nn.functional.softplus(log_scale) + 1e-6
        
#         # ============================================================
#         # Step 3: Sample with Feature-Level and Patient-Level Adaptation
#         # ============================================================
#         if noise is None:
#             noise = torch.randn(B, T, self.feature_dim, device=z_next.device)
        
#         # 🔥 应用per-feature noise scale
#         # noise_scale_per_feature: [1, feature_dim, 1]
#         # 需要调整维度匹配: [B, T, feature_dim]
#         noise_scaled = noise * self.noise_scale_per_feature.squeeze(-1)  # [B, T, feature_dim]
        
#         # 组合所有scale: context-dependent * patient-level
#         total_scale = x_scale * patient_std_scale
        
#         # 生成样本
#         x_gen = x_mean + total_scale * noise_scaled
        
#         # ============================================================
#         # 🔥 Step 4: Apply Feature-Level Scale & Bias
#         # ============================================================
#         # 让每个特征能够独立学习自己的缩放
#         x_gen = x_gen * self.feature_scale + self.feature_bias
        
#         # ============================================================
#         # 🔥 Step 5: Apply Patient-Level Mean Offset
#         # ============================================================
#         x_gen = x_gen + patient_mean_offset
        
#         return x_gen
#         # ============================================================
#         # Step 1: Mean prediction
#         # ============================================================
#         h = self.input_proj(z_next)  # [B, T, hidden_dim]
        
#         # Residual blocks
#         for block in self.res_blocks:
#             h = h + block(h)  # Residual connection
        
#         # 输出
#         x_mean = self.output_proj(h)  # [B, T, feature_dim]
        
#         # 应用可学习的尺度和偏置
#         x_mean = x_mean * self.output_scale + self.output_bias
        
#         # ============================================================
#         # Step 2: Scale prediction (context-dependent)
#         # ============================================================
#         # Scale 依赖于 Z
#         scale_delta = self.scale_net(z_next)  # [B, T, feature_dim]
        
#         # 总 scale = exp(log_base + delta)，确保正数
#         # 使用 softplus 更稳定
#         log_scale = self.log_scale_base + 0.1 * scale_delta  # delta 贡献较小
#         x_scale = torch.nn.functional.softplus(log_scale) + 1e-6
        
#         # ============================================================
#         # Step 3: Sample
#         # ============================================================
#         if noise is None:
#             noise = torch.randn_like(x_mean)
        
#         x_gen = x_mean + x_scale * noise
        
#         return x_gen
##############################<<<<<<<<<<<<<2026-January-9

#######----currently, successful version of f_x
class SSM_Out_Fx(nn.Module):
    def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0):
        super().__init__()
        
        # ============================================================
        # 1. Linear Shortcut (线性高速公路)
        # ============================================================
        # 对应传统因子模型: X = Loading_Matrix * Z
        # bias=False, 因为 bias 通常由 Data Mean 处理，或者由非线性部分拟合
        self.linear_shortcut = nn.Linear(keep_dim, feature_dim, bias=True)
        # 初始化：
        # 不同于 G (它是方阵可以用 Identity)，这里通常是矩形矩阵
        # Orthogonal 初始化有助于保持梯度流，或者 Xavier
        nn.init.orthogonal_(self.linear_shortcut.weight)
        # 1. Trend Prediction (保持不变)
        self.z_projector = nn.Sequential(
            nn.LayerNorm(keep_dim), # Pre-Norm
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim)
        )
        
        # 2. Context Encoder (保持不变)
        self.context_encoder = nn.Linear(keep_dim, d_model)
        
        # 3. Noise Modulator (保持不变)
        self.noise_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model) 
        )
        
        # 4. Residual Decoder (稍微减小容量)
        # 建议把中间层维度减半 (d_model -> d_model // 2)
        # 限制网络拟合"过度复杂噪声"的能力，迫使它更老实
        # residual_dim = d_model // 2 
        residual_dim = d_model
        
        self.context_projection = nn.Linear(d_model, residual_dim) # 适配维度
        
        self.correction_net = nn.Sequential(
            nn.Linear(residual_dim, residual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_dim, feature_dim)
        )
        
        # Zero-Init (保持不变)
        nn.init.constant_(self.correction_net[-1].weight, 0.02)
        nn.init.constant_(self.correction_net[-1].bias, 0.0)

        # ============================================================
        # 【新增】噪声阀门 (Noise Gate)
        # ============================================================
        # 一个可学习参数，控制噪声的整体幅度
        # self.noise_scale_param = nn.Parameter(torch.zeros(feature_dim))
        # 建议初始化为 -5.0 或更小
# Sigmoid(-5) ≈ 0.006 (接近关闭)
        # self.noise_scale_param = nn.Parameter(torch.full((feature_dim,), -5.0))
        
        # 硬约束：观测噪声最多只能占到数据幅度的 0.3 (根据数据归一化情况调整)
        # 如果数据已经归一化(0,1)或(-1,1)，0.2~0.3 是个合理的观测噪声上限
        # self.max_noise_scale = 5
# ============================================================
        # 【替换】通用型参数定义 (Log-Space)
        # ============================================================
        # 1. 移除了 self.max_noise_scale (不再需要人为设限)
        # 2. 参数含义变为 log(scale)
        self.log_noise_scale = nn.Parameter(torch.zeros(feature_dim))
        
        # 3. 初始化：-3.0 对应 exp(-3) ≈ 0.05
        # 无论数据多大，相对波动的起始点都是小的，比较安全。
        nn.init.constant_(self.log_noise_scale, -3.0)

    def forward(self, z_next, noise=None):
        x_linear = self.linear_shortcut(z_next)
        # Step 1: 
        x_base = self.z_projector(z_next)
        
        # Step 2: Context
        if noise is None:
            noise = torch.randn_like(x_base)
            
        context_feat = self.context_encoder(z_next)
        
        # FiLM Modulator
        noise_params = self.noise_encoder(noise)
        gamma, beta = noise_params.chunk(2, dim=-1)
        
        modulated_feat = context_feat * (1 + gamma) + beta
        
        # 降维进 Residual Net
        modulated_feat = self.context_projection(modulated_feat)
        
        # Step 3: Decode Residual
        x_residual_raw = self.correction_net(modulated_feat)
        
        # Step 4: Gated Combination (带阀门叠加)
        # scale 范围被锁死在 [0, max_noise_scale] 之间
        # real_scale = torch.sigmoid(self.noise_scale_param) * self.max_noise_scale
        
        # x_gen = x_base + real_scale * x_residual_raw
# ============================================================
        # 【替换】通用型计算 (Exp)
        # ============================================================
        # 使用 exp() 将参数映射到 (0, +inf)
        # 如果数据是 1000，模型会自动把 scale 学到 100
        # 如果数据是 0.1，模型会自动把 scale 学到 0.01
        scale = torch.exp(self.log_noise_scale) 
        
        x_gen = x_linear + x_base + scale * x_residual_raw
        
        return x_gen
### fo---backup version of f_o ------成功版本
class SSM_Out_Fo(nn.Module):
    def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0):
        super().__init__()
    
        # 1. Trend Prediction (Base Logits)
        # 负责预测由 Z 决定的"结构性缺失"模式 (例如：传感器故障往往是连续的)
        self.z_projector = nn.Sequential(
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model), 
            nn.GELU(),
            nn.Linear(d_model, feature_dim) # 输出 Logits
        )
        
        # 2. Context Encoder (保持不变)
        self.context_encoder = nn.Linear(keep_dim, d_model)
        
        # 3. Noise Modulator (保持不变)
        self.noise_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model) 
        )
        
        # 4. Residual Decoder (Bottleneck)
        # 负责预测"随机缺失"的扰动 (MCAR noise)
        residual_dim = d_model // 2 
        
        self.context_projection = nn.Linear(d_model, residual_dim)
        
        self.correction_net = nn.Sequential(
            nn.Linear(residual_dim, residual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_dim, feature_dim) # 输出 Residual Logits
        )
        
        # Zero-Init (保持不变)
        # 初始阶段 Mask 完全由 Z 决定
        nn.init.constant_(self.correction_net[-1].weight, 0.0)
        nn.init.constant_(self.correction_net[-1].bias, 0.0)

        # ============================================================
        # 【新增】噪声阀门 (Noise Gate)
        # ============================================================
        # self.noise_scale_param = nn.Parameter(torch.zeros(feature_dim))
        # 建议初始化为 -5.0 或更小
        # Sigmoid(-5) ≈ 0.006 (接近关闭)
        self.noise_scale_param = nn.Parameter(torch.full((feature_dim,), -5.0))
        
        # 对于 Logits 来说，0.5 的波动大约对应概率在 [0.38, 0.62] 之间的晃动
        # (Sigmoid(0)=0.5, Sigmoid(0.5)=0.62)
        # 这是一个安全的范围，防止随机生成的 Mask 过于混乱
        self.max_noise_scale = 0.5

    def forward(self, z_next, noise=None):
        # Step 1: Trend (Logits)
        logits_base = self.z_projector(z_next)
        
        # Step 2: Context
        if noise is None:
            # Mask 的噪声也假设为标准正态分布
            noise = torch.randn_like(logits_base)
            
        context_feat = self.context_encoder(z_next)
        
        # Step 3: FiLM Modulation
        noise_params = self.noise_encoder(noise)
        gamma, beta = noise_params.chunk(2, dim=-1)
        
        modulated_feat = context_feat * (1 + gamma) + beta
        modulated_feat = self.context_projection(modulated_feat)
        
        # Step 4: Decode Residual (Logits)
        logits_residual = self.correction_net(modulated_feat)
        
        # Step 5: Gated Combination (带阀门叠加)
        real_scale = torch.sigmoid(self.noise_scale_param) * self.max_noise_scale
        
        # Logits 叠加
        logits_final = logits_base + real_scale * logits_residual
        
        # Step 6: Sigmoid 激活 -> 得到概率 O
        o_gen = torch.sigmoid(logits_final)
        
        return o_gen


##### january-9th-2026
##### january-9th-2026
##### january-9th-2026
##### january-9th-2026
class SSM_Out_Fo_MLP(nn.Module):
    """
    Non-linear Mask Generator (Bernoulli Distribution)
    架构设计与 Fx MLP 保持一致：Early Fusion + Patient Adapter
    
    Theory: O_{t+1} ~ Bernoulli( Sigmoid( out(Z, noise) ) )
    """
    def __init__(self, feature_dim, keep_dim, d_model, zero_z=False, all_one=False):
        super().__init__()
        
        self.zero_z = zero_z  # If True, replace Z with zeros (same params, no Z-dependency)
        self.all_one = all_one  # If True, output all ones (same params/init, mask always = 1)
        self.noise_dim = feature_dim
        
        # ======================================================
        # 1. Pattern Generator (负责 Mask 之间的结构性关联)
        # ======================================================
        # 比如："如果测了指标A，通常也会测指标B"
        # Input: Z (context) + Noise
        self.input_proj = nn.Linear(keep_dim + self.noise_dim, d_model)
        
        # 使用更深的网络来捕捉变量间的采样相关性
        self.backbone = nn.Sequential(
            ResMLPBlock(d_model),
            ResMLPBlock(d_model) # Mask 任务比数值预测简单，两层通常够了
        )
        
        # 输出 Logits 的形状 (未经过 Sigmoid)
        self.head_pattern = nn.Linear(d_model, feature_dim)
        
        # ======================================================
        # 2. Patient Adapter (负责该病人的采样频率基准)
        # ======================================================
        # 比如：ICU 病人采样率高 (Bias 大)，普通病房采样率低 (Bias 小)
        self.patient_adapter = nn.Sequential(
            nn.Linear(keep_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, feature_dim * 2) # 输出 [scale, bias] for logits
        )
        
        # 🔥 全局输出偏置 (用于自动初始化匹配数据的 obs_rate)
        self.output_bias = nn.Parameter(torch.zeros(feature_dim))
        
        self._init_specific_weights()

    def _init_specific_weights(self):
        # 专门针对 Mask 任务的初始化
        # 1. Mask 通常是稀疏的 (0 多 1 少)
        # 所以 Adapter 的 Bias 初始化为负数，让 sigmoid 输出偏向 0
        nn.init.constant_(self.patient_adapter[-1].bias, -2.0) 
        
        # 2. Pattern Head 初始化为接近 0，避免初始噪声太大导致全是 0.5
        nn.init.xavier_uniform_(self.head_pattern.weight, gain=0.1)
        nn.init.zeros_(self.head_pattern.bias)
        
        # 3. output_bias 初始化为 0 (会被训练代码自动设置为 logit(obs_rate))
        nn.init.zeros_(self.output_bias)

    def forward(self, z_next, noise=None):
        """
        Args:
            z_next: [B, T, keep_dim]
            noise:  [B, T, feature_dim] (Optional)
        Returns:
            o_prob: [B, T, feature_dim] 范围 (0, 1) 的概率
        """
        B, T, _ = z_next.shape
        
        # If zero_z or all_one mode: replace Z with zeros
        # This keeps the same params and random state as fo_type='mlp'
        if self.zero_z or self.all_one:
            z_next = torch.zeros_like(z_next)
        
        # ---------------------------------------------------
        # A. 计算 Patient-Specific Logit Statistics
        # ---------------------------------------------------
        # adapter_out: 决定了这个时刻、这个病人、每个变量的 Logit 分布参数
        adapter_out = self.patient_adapter(z_next) 
        logit_scale, logit_bias = adapter_out.chunk(2, dim=-1)
        
        # Scale (必须 > 0): 控制噪声对概率的影响程度
        # 如果 Scale 大 -> 概率在 0 和 1 两端跳跃 (确定性强)
        # 如果 Scale 小 -> 概率在基准线附近微调
        logit_scale = F.softplus(logit_scale) + 0.1 
        
        # ---------------------------------------------------
        # B. 生成 Dynamic Logit Pattern (Early Fusion)
        # ---------------------------------------------------
        if noise is None:
            # 这里的噪声经过 MLP 混合，可以捕捉 "同时采样" 的相关性
            noise = torch.randn(B, T, self.noise_dim, device=z_next.device)
            
        combined = torch.cat([z_next, noise], dim=-1)
        h = self.backbone(self.input_proj(combined))
        
        raw_logits = self.head_pattern(h) # Center around 0
        
        # ---------------------------------------------------
        # C. 融合与激活
        # ---------------------------------------------------
        # 1. 仿射变换得到最终 Logits
        # 🔥 添加全局 output_bias (用于匹配数据的 obs_rate)
        final_logits = raw_logits * logit_scale + logit_bias + self.output_bias
        
        # 2. Sigmoid 映射到概率空间
        o_prob = torch.sigmoid(final_logits)
        
        # If all_one mode: discard the computation and return constant ones
        # (full forward was still run to consume the same random state as mlp)
        if self.all_one:
            return torch.ones_like(o_prob)
        
        return o_prob


class SSM_Out_Fo_Oracle(nn.Module):
    """
    Oracle Mask Generator that matches the true DGP:
        P(O_{t,j}=1 | Z_t) = σ(W @ Z_t + b)
    
    Matches the case_info DGP where:
        P(O_j=1) = σ(c + β * b_j * v_t)
    
    The linear layer W (p x d) can learn the sparse structure
    (only the v-column active) and b learns the calibrated intercept c.
    
    - Learnable: W (p x d), b (p,)
    - No noise injection, no MLP backbone, no patient adapter
    - Clean, low-variance gradients on Z (linear in Z → sigmoid)
    """
    def __init__(self, feature_dim, keep_dim, d_model=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        
        # W_mask: (p, d) linear projection Z → logits
        self.W_mask = nn.Linear(keep_dim, feature_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Small init for W to avoid dominating Z gradients early on
        nn.init.xavier_uniform_(self.W_mask.weight, gain=0.5)
        # b_mask init to 0 (will be learned to approximate calibrated intercept c)
        nn.init.zeros_(self.W_mask.bias)
    
    def forward(self, z_next, noise=None):
        """
        Args:
            z_next: [B, T, keep_dim]
            noise: ignored (kept for interface compatibility)
        Returns:
            o_prob: [B, T, feature_dim] in (0, 1)
        """
        # logits = Z @ W^T + b
        logits = self.W_mask(z_next)  # (B, T, p)
        
        # o_prob = σ(logits)
        o_prob = torch.sigmoid(logits)
        
        return o_prob


class SSM_Out_Fo_OracleFixed(nn.Module):
    """
    Oracle Mask Generator initialized (optionally frozen) to ground truth DGP:
        P(O_{t,j}=1 | Z_t) = σ(W @ Z_t + b)
    
    W and b are set from the true DGP parameters.
    If freeze=True, W and b are not trainable (pure oracle).
    If freeze=False, W and b start at ground truth but can be fine-tuned.
    """
    def __init__(self, feature_dim, keep_dim, d_model=None,
                 gt_weight=None, gt_bias=None, freeze=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        
        self.W_mask = nn.Linear(keep_dim, feature_dim)
        
        if gt_weight is not None:
            # gt_weight: (feature_dim, keep_dim)
            self.W_mask.weight.data.copy_(gt_weight)
        if gt_bias is not None:
            # gt_bias: (feature_dim,)
            self.W_mask.bias.data.copy_(gt_bias)
        
        if freeze:
            self.W_mask.weight.requires_grad_(False)
            self.W_mask.bias.requires_grad_(False)
    
    def forward(self, z_next, noise=None):
        logits = self.W_mask(z_next)  # (B, T, p)
        o_prob = torch.sigmoid(logits)
        return o_prob


class SSM_Out_Fo_Blend(nn.Module):
    """
    Blended Mask Generator — matches the data_gen_stren DGP structure:

        P(O_{j,t}=1 | Z_t) = alpha * f_o^{MLP}(Z_t)_j  +  (1-alpha) * f'_o(Z_t)_j

    The analytical component f'_o has a **learnable** severity-driven form:
        f'_o(Z_t)_j = σ( c_j  +  β_j · Z_{k*,t} )

    where
        k*  = severity_dim  (integer index, fixed — the Z dimension with highest
                             temporal variance; discrete so not gradient-able)
        β   ∈ R^p           (learnable per-feature severity sensitivity)
        c   ∈ R^p           (learnable per-feature logit intercept)

    Initialization uses the DGP values (beta=2.5, c=c_intercept, b_signs) so the
    model starts exactly at the data-generating structure and can fine-tune from there.

    alpha is a fixed buffer (not trainable — it is a design choice, not a parameter):
        alpha=0  →  pure analytical f'_o (MLP output is ignored)
        alpha=1  →  pure MLP f_o  (equivalent to SSM_Out_Fo_MLP)
        0<alpha<1 →  blend
    """

    def __init__(self, feature_dim, keep_dim, d_model,
                 alpha=0.0, beta=2.5, c_intercept=0.0, severity_dim=0,
                 b_vec=None):
        """
        Args:
            feature_dim   : p
            keep_dim      : d (latent dimension)
            d_model       : hidden size forwarded to fo_mlp
            alpha         : fixed blend weight (0=pure f'_o, 1=pure MLP)
            beta          : initial value for learnable β (scalar broadcast to p)
            c_intercept   : initial value for learnable c (scalar broadcast to p)
            severity_dim  : fixed index of the severity Z-dimension (not trained)
            b_vec         : (p,) initial sign vector; default first-half +1, second-half −1.
                            Values are used as initial β_j = beta * b_j, giving each
                            feature its own signed slope from the start.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        self.severity_dim = int(severity_dim)   # fixed integer index

        # ── Learnable MLP component ───────────────────────────────────────────
        self.fo_mlp = SSM_Out_Fo_MLP(feature_dim, keep_dim, d_model)

        # ── Learnable f'_o parameters ─────────────────────────────────────────
        # beta_j : per-feature severity sensitivity, init = beta * b_j
        if b_vec is None:
            b = torch.ones(feature_dim)
            b[feature_dim // 2:] = -1.0
        else:
            b = torch.as_tensor(b_vec, dtype=torch.float32).clone()

        # β_j = beta * b_j  (preserves sign structure at init)
        self.beta_vec = nn.Parameter((float(beta) * b).clone())   # (p,)

        # c_j : per-feature intercept, init = c_intercept (same for all)
        self.c_vec = nn.Parameter(
            torch.full((feature_dim,), float(c_intercept))        # (p,)
        )

        # alpha: fixed blend control — not a parameter
        self.register_buffer('alpha_val', torch.tensor(float(alpha), dtype=torch.float32))

    # ------------------------------------------------------------------
    def _fo_prime(self, z_next):
        """
        Learnable analytical f'_o(Z).
        Args:    z_next : (B, T, d)
        Returns: probs  : (B, T, p) in (0, 1)
        """
        z_k = z_next[:, :, self.severity_dim]                          # (B, T)
        # logits[b,t,j] = c_j + beta_j * z_k[b,t]
        logits = (self.c_vec.view(1, 1, -1)
                  + self.beta_vec.view(1, 1, -1) * z_k.unsqueeze(-1)) # (B, T, p)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    def forward(self, z_next, noise=None):
        """
        Args:
            z_next : (B, T, d)
            noise  : optional noise tensor (forwarded to fo_mlp; ignored for f'_o)
        Returns:
            o_prob : (B, T, p) in (0, 1)
        """
        alpha = self.alpha_val.item()

        if alpha == 0.0:
            # Pure analytical — skip MLP computation entirely
            return self._fo_prime(z_next)
        elif alpha == 1.0:
            # Pure MLP — skip analytical computation entirely
            return self.fo_mlp(z_next, noise)
        else:
            p_mlp   = self.fo_mlp(z_next, noise)    # (B, T, p)
            p_prime = self._fo_prime(z_next)         # (B, T, p)
            return alpha * p_mlp + (1.0 - alpha) * p_prime


#############----------------------- simulation version
# class SSM_Out_Fo_MLP(nn.Module):
#     """
#     MLP-only mask decoder for nonlinear mask prediction.
    
#     设计理念（模仿 SSM_Out_Fx_MLP，但输出是 mask 概率）：
#     1. 纯 MLP 学习非线性映射 Z → O (mask probability)
#     2. 使用 Residual MLP 块，更深更宽，能学习复杂非线性
#     3. Noise 处理：context-dependent scale + 加性噪声
#        - mean_net: 预测 E[logit(O)|Z]
#        - scale_net: 预测 Std[logit(O)|Z]
#     4. 最后用 Sigmoid 输出概率
#     """
#     def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0):
#         super().__init__()
        
#         self.feature_dim = feature_dim
#         self.keep_dim = keep_dim
        
#         # 更宽的隐藏层
#         hidden_dim = max(d_model * 2, 64)  # 比 Fx 稍小，mask 更简单
        
#         # ============================================================
#         # 1. Mean Network: Z → E[logit(O)|Z]
#         # ============================================================
#         self.input_proj = nn.Linear(keep_dim, hidden_dim)
        
#         # Residual MLP blocks (比 Fx 浅一些，mask 不需要太复杂)
#         self.res_blocks = nn.ModuleList([
#             self._make_res_block(hidden_dim, dropout) for _ in range(2)
#         ])
        
#         # 输出层 (输出 logits)
#         self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
#         # 输出尺度缩放（可学习）
#         self.output_scale = nn.Parameter(torch.ones(feature_dim))
#         # Bias 初始化为 0，对应 sigmoid(0) = 0.5 的 prior
#         self.output_bias = nn.Parameter(torch.zeros(feature_dim))
        
#         # ============================================================
#         # 2. Scale Network: Z → Std[logit(O)|Z] (context-dependent noise)
#         # ============================================================
#         self.scale_net = nn.Sequential(
#             nn.Linear(keep_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 2, feature_dim)
#         )
        
#         # Scale 的基线（log space，初始化为较小值，mask 不需要太大噪声）
#         self.log_scale_base = nn.Parameter(torch.full((feature_dim,), -3.0))
        
#         # 最大噪声尺度限制（防止 mask 过于随机）
#         self.max_scale = 0.5
        
#         # ============================================================
#         # 3. 初始化
#         # ============================================================
#         self._init_weights()
    
#     def _make_res_block(self, hidden_dim, dropout):
#         """创建一个 Residual MLP 块"""
#         return nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Dropout(dropout)
#         )
    
#     def _init_weights(self):
#         """初始化权重"""
#         # 输入投影
#         nn.init.xavier_uniform_(self.input_proj.weight)
#         nn.init.zeros_(self.input_proj.bias)
        
#         # Residual blocks - 最后一层小初始化
#         for block in self.res_blocks:
#             for module in block:
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
#             # 最后一层小初始化，保持 residual 性质
#             nn.init.normal_(block[-2].weight, std=0.02)
        
#         # 输出投影 - 小初始化，让初始输出接近 0 (对应 0.5 概率)
#         nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
#         nn.init.zeros_(self.output_proj.bias)
        
#         # Scale network - 小初始化
#         for module in self.scale_net:
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight, gain=0.1)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)

#     def forward(self, z_next, noise=None):
#         """
#         Args:
#             z_next: [B, T, keep_dim] 隐状态
#             noise: [B, T, feature_dim] 可选的外部噪声
#         Returns:
#             o_gen: [B, T, feature_dim] 生成的 mask 概率 (0-1)
#         """
#         # ============================================================
#         # Step 1: Mean prediction (logits)
#         # ============================================================
#         h = self.input_proj(z_next)  # [B, T, hidden_dim]
        
#         # Residual blocks
#         for block in self.res_blocks:
#             h = h + block(h)  # Residual connection
        
#         # 输出 logits
#         logits_mean = self.output_proj(h)  # [B, T, feature_dim]
        
#         # 应用可学习的尺度和偏置
#         logits_mean = logits_mean * self.output_scale + self.output_bias
        
#         # ============================================================
#         # Step 2: Scale prediction (context-dependent)
#         # ============================================================
#         scale_delta = self.scale_net(z_next)  # [B, T, feature_dim]
        
#         # 总 scale = sigmoid(log_base + delta) * max_scale
#         # 限制在 [0, max_scale] 范围内
#         log_scale = self.log_scale_base + 0.1 * scale_delta
#         logits_scale = torch.sigmoid(log_scale) * self.max_scale
        
#         # ============================================================
#         # Step 3: Sample and Sigmoid
#         # ============================================================
#         if noise is None:
#             noise = torch.randn_like(logits_mean)
        
#         # 在 logit space 加噪声
#         logits_final = logits_mean + logits_scale * noise
        
#         # Sigmoid 转换为概率
#         o_gen = torch.sigmoid(logits_final)
        
#         return o_gen
#########################<<<<<<< simulation version
# import torch
# import torch.nn as nn


class SSM_Out_Fo_Constant(nn.Module):
    """
    Constant Mask Generator (Baseline for ablation study).
    
    f_o(z) = constant, i.e., the mask prediction does NOT depend on Z.
    This removes informative missingness: the observation pattern carries
    no information about the latent state.
    
    Output: O ~ Bernoulli(Sigmoid(bias))
    where bias is a learnable per-feature parameter (not conditioned on Z).
    Noise is still injected so each sample gets different masks.
    """
    def __init__(self, feature_dim, keep_dim, d_model, dropout=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Learnable per-feature logit bias (initialized to 0 → prob=0.5)
        # Will be auto-calibrated to match data obs_rate via output_bias
        self.output_bias = nn.Parameter(torch.zeros(feature_dim))
        
        # Small noise scale so each sample gets slightly different masks
        # but the *expected* mask is constant across all Z
        self.noise_scale = nn.Parameter(torch.full((feature_dim,), -3.0))

    def forward(self, z_next, noise=None):
        """
        Args:
            z_next: [B, T, keep_dim] - latent state (IGNORED)
            noise:  [B, T, feature_dim] (Optional)
        Returns:
            o_prob: [B, T, feature_dim] in (0, 1)
        """
        B, T, _ = z_next.shape
        
        if noise is None:
            noise = torch.randn(B, T, self.feature_dim, device=z_next.device)
        
        # Constant logits: only depends on learnable bias + noise
        scale = torch.sigmoid(self.noise_scale)  # (0, 1)
        logits = self.output_bias.unsqueeze(0).unsqueeze(0) + scale * noise
        
        o_prob = torch.sigmoid(logits)
        return o_prob


class SSM_Transition_G(nn.Module):
    def __init__(self, keep_dim, d_model, dropout=0.0):
        """
        State Transition Function: Z_{t+1} = LayerNorm(A * Z_t + f(Z_t) + g(Z_t, epsilon))

        Output LayerNorm guarantees rollout stability:
        - During training the encoder constrains Z, so without output normalisation
          the joint loss gives no gradient signal enforcing autoregressive stability.
        - The learnable (gamma, beta) of LayerNorm restore expressive amplitude/scale
          information, so out_fx can still learn the correct output magnitude.
        - Spectral radius of linear_A is therefore no longer a concern for generation.
        """
        super().__init__()
        
        # ============================================================
        # 1. 线性高速公路 (Linear Highway) - 模拟 Kalman Filter 的 F 矩阵
        # ============================================================
        # 显式建模: Z_{t+1} = A @ Z_t
        # bias=False 因为我们希望 Z=0 时保持不动 (除非有外力)
        self.linear_A = nn.Linear(keep_dim, keep_dim, bias=False)
        
        # 【关键初始化】初始化为单位矩阵 (Identity)
        # 含义: 默认预测 "明天 = 今天" (Random Walk 假设)
        # 这为 Forecasting 提供了一个非常稳健的起点
        nn.init.eye_(self.linear_A.weight)
        
        # ============================================================
        # 2. 非线性动力学修正 (Non-linear Dynamics Correction)
        # ============================================================
        # 负责拟合线性模型无法捕捉的复杂流形 (例如: 饱和效应、突变)
        # 使用 Pre-Norm 结构: Norm -> Linear -> Act -> Linear
        self.trend_net = nn.Sequential(
            nn.LayerNorm(keep_dim),     # <--- Pre-Norm: 只规范化输入，不限制输出
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, keep_dim)
        )
        
        # 【Zero-Init】初始时让非线性分支输出为 0
        # 这样模型一开始就是纯线性的，随着训练慢慢学复杂的非线性
        nn.init.constant_(self.trend_net[-1].weight, 0.0)
        nn.init.constant_(self.trend_net[-1].bias, 0.0)
        
        # ============================================================
        # 3. 过程噪声融合 (Process Noise Fusion)
        # ============================================================
        # 允许噪声依赖于当前状态 (State-Dependent Noise)
        self.noise_fusion = nn.Sequential(
            nn.LayerNorm(keep_dim * 2), # <--- Pre-Norm 处理 Input (Z + Noise)
            nn.Linear(keep_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, keep_dim)
        )
        
        # 【Small-Init】过程噪声一开始不要太大，避免冲散动力学
        nn.init.normal_(self.noise_fusion[-1].weight, std=0.1)
        nn.init.constant_(self.noise_fusion[-1].bias, 0.0)

        # ============================================================
        # 4. Output LayerNorm — guarantees autoregressive rollout stability
        # ============================================================
        # During joint training the loss uses encoder Z_f (not free rollouts),
        # so there is no gradient signal enforcing stability of free generation.
        # LayerNorm here normalises every Z_t output to zero-mean/unit-var
        # across the d latent dims, making ||Z_t|| bounded by sqrt(d)*||gamma||.
        # The learnable (gamma, beta) still allow out_fx to learn any desired scale.
        self.output_norm = nn.LayerNorm(keep_dim)

    def forward(self, z_prev, noise=None):
        """
        z_prev: (..., keep_dim)  — any leading batch dims are handled (e.g. (B, d) or (B, T, d))
        noise:  same shape as z_prev, or None (samples fresh Gaussian noise)
        """
        # Step A: 线性主导 (The Backbone)
        z_linear = self.linear_A(z_prev)

        # Step B: 非线性修正 (The Correction)
        z_trend_correction = self.trend_net(z_prev)

        # Step C: 过程噪声 (The Uncertainty)
        if noise is None:
            noise = torch.randn_like(z_prev)

        fusion_input = torch.cat([z_prev, noise], dim=-1)
        z_stochastic = self.noise_fusion(fusion_input)

        # Step D: 叠加 + Output LayerNorm
        # LayerNorm bounds ||Z_t|| regardless of linear_A spectral radius,
        # solving the free-rollout divergence problem at its source.
        z_next = self.output_norm(z_linear + z_trend_correction + z_stochastic)

        return z_next


class SSM_Transition_G_KOrder(nn.Module):
    """
    K-order Markovian State Transition Function:
    Z_{t+1} = f(Z_{t-k+1}, Z_{t-k+2}, ..., Z_t) + g(epsilon)
    
    使用 GRU 聚合 k 个历史状态，然后预测下一个状态。
    对于 k=1，行为等价于 SSM_Transition_G。
    """
    def __init__(self, keep_dim, d_model, k_order=1, dropout=0.0):
        super().__init__()
        self.k_order = k_order
        self.keep_dim = keep_dim
        self.d_model = d_model
        
        if k_order > 1:
            # ============================================================
            # 1. History Aggregation: GRU to encode k previous states
            # ============================================================
            self.history_gru = nn.GRU(
                input_size=keep_dim,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            
            # Project aggregated history back to keep_dim
            self.history_proj = nn.Linear(d_model, keep_dim)
            
            # ============================================================
            # 2. Linear Highway (same as SSM_Transition_G)
            # ============================================================
            # For k-order, we use the projected aggregated state
            self.linear_A = nn.Linear(keep_dim, keep_dim, bias=False)
            nn.init.eye_(self.linear_A.weight)
        else:
            # k=1: Same as original SSM_Transition_G
            self.linear_A = nn.Linear(keep_dim, keep_dim, bias=False)
            nn.init.eye_(self.linear_A.weight)
        
        # ============================================================
        # 3. Non-linear Dynamics Correction
        # ============================================================
        self.trend_net = nn.Sequential(
            nn.LayerNorm(keep_dim),
            nn.Linear(keep_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, keep_dim)
        )
        nn.init.constant_(self.trend_net[-1].weight, 0.0)
        nn.init.constant_(self.trend_net[-1].bias, 0.0)
        
        # ============================================================
        # 4. Process Noise Fusion
        # ============================================================
        self.noise_fusion = nn.Sequential(
            nn.LayerNorm(keep_dim * 2),
            nn.Linear(keep_dim * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, keep_dim)
        )
        nn.init.normal_(self.noise_fusion[-1].weight, std=0.1)
        nn.init.constant_(self.noise_fusion[-1].bias, 0.0)

    def forward(self, z_window, noise=None):
        """
        Args:
            z_window: [B, k, keep_dim] for k>1, or [B, keep_dim] for k=1
            noise: [B, keep_dim] or None
        
        Returns:
            z_next: [B, keep_dim]
        """
        if self.k_order > 1:
            # Ensure input is [B, k, keep_dim]
            if z_window.dim() == 2:
                z_window = z_window.unsqueeze(1)  # [B, 1, keep_dim]
            
            B = z_window.shape[0]
            device = z_window.device
            
            # Handle case where window is smaller than k (beginning of sequence)
            actual_k = z_window.shape[1]
            if actual_k < self.k_order:
                # Pad with zeros or repeat first state
                padding = z_window[:, :1, :].expand(-1, self.k_order - actual_k, -1)
                z_window = torch.cat([padding, z_window], dim=1)
            
            # Aggregate history using GRU
            _, h_n = self.history_gru(z_window)  # h_n: [1, B, d_model]
            z_agg = self.history_proj(h_n.squeeze(0))  # [B, keep_dim]
            
            # Use most recent state for linear highway
            z_prev = z_window[:, -1, :]  # [B, keep_dim]
        else:
            # k=1: z_window is [B, keep_dim] or [B, 1, keep_dim]
            if z_window.dim() == 3:
                z_prev = z_window[:, -1, :]  # Take last state
            else:
                z_prev = z_window
            z_agg = z_prev
            B = z_prev.shape[0]
            device = z_prev.device
        
        # Step A: Linear highway
        z_linear = self.linear_A(z_prev)
        
        # Step B: Non-linear correction (using aggregated state for k>1)
        z_trend_correction = self.trend_net(z_agg)
        
        # Step C: Process noise
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        fusion_input = torch.cat([z_agg, noise], dim=-1)
        z_stochastic = self.noise_fusion(fusion_input)
        
        # Step D: Final output
        z_next = z_linear + z_trend_correction + z_stochastic
        
        return z_next


# ============================================================
# 🔥 增强版 SSM_Filter_V2: 借鉴 CSDI 的双向注意力设计
# ============================================================
# 解决三个问题：
# 1. 输入处理：三通道分离 (x, mask, feature_emb)
# 2. 双向注意力：时间方向 + 特征方向
# 3. 更强的架构：多层 ResidualBlock

def get_torch_trans(heads=8, layers=1, channels=64):
    """CSDI 风格的 Transformer 层"""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels * 2, activation="gelu", batch_first=False
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class BiDirectionalBlock(nn.Module):
    """
    双向注意力块 (借鉴 CSDI 的 ResidualBlock)
    
    每一层同时做：
    1. forward_time(): 时间方向的 attention (同一个特征，不同时间点之间)
       🔥 使用 causal mask，确保 z_t = h(X_{1:t}^o)，只能看当前和过去
    2. forward_feature(): 特征方向的 attention (同一个时间点，不同特征之间)
       特征方向不需要 causal mask（同一时间点的不同特征可以互相看）
    """
    def __init__(self, channels, n_heads, feature_dim, time_dim, dropout=0.1, causal=True):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.causal = causal  # 🔥 是否使用 causal mask
        
        # 时间方向 Transformer
        self.time_layer = get_torch_trans(heads=n_heads, layers=1, channels=channels)
        
        # 特征方向 Transformer
        self.feature_layer = get_torch_trans(heads=n_heads, layers=1, channels=channels)
        
        # 🔥 缓存 causal mask，避免重复创建
        self._causal_mask_cache = {}
        
        # 中间投影 (with Gated Linear Unit)
        self.mid_projection = nn.Conv1d(channels, 2 * channels, 1)
        
        # 输出投影
        self.output_projection = nn.Conv1d(channels, 2 * channels, 1)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def _get_causal_mask(self, L, device):
        """
        🔥 生成 causal mask（下三角矩阵，True = 被遮盖，不能看）
        
        对于 nn.TransformerEncoder，mask 的语义是：
        - True: 该位置被遮盖，不能 attend
        - False: 该位置可见
        
        下三角矩阵（对角线+下方=False，上方=True）确保只能看当前和过去。
        """
        cache_key = (L, device)
        if cache_key not in self._causal_mask_cache:
            # 上三角矩阵（不含对角线），True = 被遮盖
            # mask[i,j] = True if j > i (不能看未来)
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]
        
    def forward_time(self, y, base_shape):
        """
        🔥 时间方向 attention: (B, C, K, L) -> 每个特征独立地对时间做 attention
        
        使用 causal mask 确保 z_t = h(X_{1:t}^o)，只能看当前和过去！
        """
        B, channel, K, L = base_shape
        if L == 1:
            return y
        
        # (B, C, K*L) -> (B, K, C, L) -> (B*K, C, L) -> (L, B*K, C)
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = y.permute(2, 0, 1)  # (L, B*K, C) - Transformer expects (L, N, C)
        
        # 🔥 关键修复：添加 causal mask
        if self.causal:
            causal_mask = self._get_causal_mask(L, y.device)
            y = self.time_layer(y, mask=causal_mask)
        else:
            y = self.time_layer(y)
        
        y = y.permute(1, 2, 0)  # (B*K, C, L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y
    
    def forward_feature(self, y, base_shape):
        """特征方向 attention: (B, C, K, L) -> 每个时间点独立地对特征做 attention"""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        
        # (B, C, K*L) -> (B, L, C, K) -> (B*L, C, K) -> (K, B*L, C)
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward(self, x):
        """
        x: (B, channels, K, L) where K=feature_dim, L=time_dim
        """
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        
        # 双向 attention
        y = self.forward_time(x, base_shape)
        y = self.forward_feature(y, base_shape)
        
        # Gated Linear Unit
        y = self.mid_projection(y)
        gate, filter_out = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_out)
        
        # Output projection with skip connection
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        return (x + residual) / math.sqrt(2.0), skip


# ============================================================
# 🔥 Continuous Sinusoidal Time Embedding (连续正弦时间编码)
# ============================================================
# 用于不规则采样的医疗时序数据 (PhysioNet)
# 与 CSDI 保持一致，使用连续时间戳而非离散整数索引
# ============================================================

class ContinuousTimeEmbedding(nn.Module):
    """
    连续正弦时间编码 (Continuous Sinusoidal Time Embedding)
    
    适用于不规则采样数据:
    - 输入: 连续时间戳 (可以是小时、分钟等)
    - 输出: 时间编码向量
    
    与 CSDI 一致的实现:
    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 预计算除数项 (与 CSDI 一致)
        # div_term = 1 / 10000^(2i/d)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).float() / d_model
        )
        self.register_buffer('div_term', div_term)
    
    def forward(self, time_stamps):
        """
        Args:
            time_stamps: (B, T) 或 (T,) 连续时间戳
        
        Returns:
            pe: (B, T, d_model) 或 (T, d_model) 时间编码
        """
        # 确保输入至少是 2D
        squeeze_batch = False
        if time_stamps.dim() == 1:
            time_stamps = time_stamps.unsqueeze(0)
            squeeze_batch = True
        
        B, T = time_stamps.shape
        device = time_stamps.device
        
        # 初始化输出
        pe = torch.zeros(B, T, self.d_model, device=device)
        
        # position: (B, T, 1)
        position = time_stamps.unsqueeze(-1)
        
        # div_term: (d_model/2,) -> (1, 1, d_model/2)
        div_term = self.div_term.unsqueeze(0).unsqueeze(0)
        
        # 计算 sin 和 cos
        pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        if squeeze_batch:
            pe = pe.squeeze(0)
        
        return pe


##### new version--- january-4-2026
class SSM_Filter_V2(nn.Module):
    """
    增强版 SSM_Filter，借鉴 CSDI 的设计：
    
    改进点：
    1. 三通道输入：[x, mask, feature_embedding] 而不是简单拼接
    2. 双向注意力：时间方向 + 特征方向 (每层都有)
    3. 可学习的 Feature Embedding (类似 CSDI 的 side_info)
    4. 更深的架构：多个 BiDirectionalBlock
    5. 🔥 连续正弦时间编码：支持不规则采样 (与 CSDI 一致)
    """
    def __init__(
        self,
        time_dim,
        feature_dim,
        keep_dim,
        d_model,
        d_inner, n_head, d_k, d_v, dropout, n_groups, n_group_inner_layers, **kwargs
    ):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.keep_dim = keep_dim
        self.d_model = d_model
        self.n_groups = n_groups
        
        # ============================================================
        # 1. 输入处理：三通道分离 (借鉴 CSDI)
        # ============================================================
        # 输入维度: 2 (noisy_x + mask) -> 先投影到 d_model
        self.input_projection = nn.Conv1d(2, d_model, 1)
        nn.init.kaiming_normal_(self.input_projection.weight)
        
        # ============================================================
        # 2. Feature Embedding (可学习，类似 CSDI 的 feature 方向 position)
        # ============================================================
        # 每个特征有自己的 "身份证"
        self.feature_embedding = nn.Embedding(feature_dim, d_model)
        
        # 🔥 [NEW] Continuous Sinusoidal Time Embedding
        # 适用于不规则采样的医疗数据 (PhysioNet)
        self.time_embedding = ContinuousTimeEmbedding(d_model)
        
        # ============================================================
        # 3. 双向注意力堆叠
        # 🔥 默认 causal=True，确保 z_t = h(X_{1:t}^o)
        # ============================================================
        self.blocks = nn.ModuleList([
            BiDirectionalBlock(
                channels=d_model,
                n_heads=n_head,
                feature_dim=feature_dim,
                time_dim=time_dim,
                dropout=dropout,
                causal=True  # 🔥 关键！时间方向使用 causal mask
            ) for _ in range(n_groups)
        ])
        
        # ============================================================
        # 4. 输出投影 (直接从 d_model 映射到 keep_dim，避免信息丢失)
        # ============================================================
        self.output_projection1 = nn.Conv1d(d_model, d_model, 1)
        
        # 🔥 修复: 直接从 d_model 映射到 keep_dim，不再压缩到 1 通道
        self.head_trend = nn.Linear(d_model, keep_dim)
        
        # ============================================================
        # 5. Mask-Aware Residual (保留原有设计)
        # ============================================================
        self.mask_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU()
        )
        
        self.noise_fusion = nn.Sequential(
            nn.Linear(2 * d_model + keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.head_residual = nn.Linear(d_model, keep_dim)
        nn.init.normal_(self.head_residual.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.head_residual.bias, 0.0)

    def forward(self, x, mask, time_stamps=None):
        """
        Args:
            x: (B, T, p) 观测数据
            mask: (B, T, p) 观测 mask
            time_stamps: (B, T) 或 (T,) 或 None
                - 连续时间戳 (小时, 分钟等)
                - 如果为 None，使用默认的整数序列 [0, 1, 2, ..., T-1]
        """
        B, T, p = x.shape
        device = x.device
        
        # ============================================================
        # Step 1: 输入处理 (三通道)
        # ============================================================
        mask_float = mask.float()
        
        # 缺失位置填充噪声
        x_noisy = x * mask_float + (1 - mask_float) * torch.randn_like(x)
        
        # 拼接 [x_noisy, mask] -> (B, T, p, 2) -> (B, 2, p, T)
        x_input = torch.stack([x_noisy, mask_float], dim=-1)  # (B, T, p, 2)
        x_input = x_input.permute(0, 3, 2, 1)  # (B, 2, p, T) = (B, inputdim, K, L)
        
        # 投影到 d_model: (B, 2, p*T) -> (B, d_model, p*T)
        x_flat = x_input.reshape(B, 2, p * T)
        x_proj = self.input_projection(x_flat)
        x_proj = F.relu(x_proj)
        x_proj = x_proj.reshape(B, self.d_model, p, T)  # (B, C, K, L)
        
        # ============================================================
        # Step 2: 添加 Embedding (Feature + Time)
        # ============================================================
        # Feature embedding: (p,) -> (p, d_model) -> (1, d_model, p, 1)
        feat_emb = self.feature_embedding(torch.arange(p, device=device))
        feat_emb = feat_emb.T.unsqueeze(0).unsqueeze(-1)  # (1, d_model, p, 1)
        
        # 🔥 [NEW] Continuous Sinusoidal Time Embedding
        # 支持不规则采样的连续时间戳
        if time_stamps is None:
            # 默认使用整数序列 [0, 1, 2, ..., T-1]
            time_stamps = torch.arange(T, dtype=torch.float32, device=device)
        
        # 确保 time_stamps 形状正确
        if time_stamps.dim() == 1:
            # (T,) -> broadcast to all samples
            time_emb = self.time_embedding(time_stamps)  # (T, d_model)
            time_emb = time_emb.T.unsqueeze(0).unsqueeze(2)  # (1, d_model, 1, T)
        else:
            # (B, T) -> per-sample time stamps
            time_emb = self.time_embedding(time_stamps)  # (B, T, d_model)
            time_emb = time_emb.permute(0, 2, 1).unsqueeze(2)  # (B, d_model, 1, T)
        
        # 叠加 embedding
        x_proj = x_proj + feat_emb + time_emb
        
        # ============================================================
        # Step 3: 双向注意力 (多层)
        # ============================================================
        skip_connections = []
        for block in self.blocks:
            x_proj, skip = block(x_proj)
            skip_connections.append(skip)
        
        # 聚合 skip connections (CSDI 风格)
        x_out = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.blocks))
        
        # ============================================================
        # Step 4: 输出投影
        # ============================================================
        # (B, d_model, p, T) -> (B, d_model, p*T)
        x_out = x_out.reshape(B, self.d_model, p * T)
        x_out = self.output_projection1(x_out)
        x_out = F.relu(x_out)
        
        # 🔥 修复: 直接从 d_model 映射到 keep_dim，不再压缩到 1 通道
        # 对 feature 维度做 mean pooling: (B, d_model, p*T) -> (B, d_model, p, T) -> (B, d_model, T)
        Z_feat = x_out.reshape(B, self.d_model, p, T).mean(dim=2)  # (B, d_model, T)
        Z_feat = Z_feat.permute(0, 2, 1)  # (B, T, d_model)
        
        # ============================================================
        # Step 5: 输出生成 (Mask-Aware)
        # ============================================================
        # 🔥 修复: 直接从 d_model 映射到 keep_dim
        z_trend = self.head_trend(Z_feat)  # (B, T, keep_dim)
        
        # Mask-aware residual
        mask_feat = self.mask_proj(mask_float)  # (B, T, d_model)
        noise = torch.randn(B, T, self.keep_dim, device=device)
        fusion_input = torch.cat([Z_feat, noise, mask_feat], dim=-1)
        noise_feat = self.noise_fusion(fusion_input)
        z_residual = self.head_residual(noise_feat)
        
        Z_final = z_trend + z_residual
        
        return Z_final


class SSM_Smoother_V2(nn.Module):
    """
    增强版 SSM_Smoother，加入更强的时序归纳偏置：
    
    改进点：
    1. 双向 GRU 提取时序特征 (而不是简单的 cross-attention)
    2. 保留 attention 机制用于长程依赖
    3. 加入 local window attention (时序局部性)
    4. 🔥 预计算机制：H 的特征只计算一次，避免 O(T^2) 复杂度
    5. 🔥 Delta 预测 + Residual Connection：
       - z_t = z_{t+1} + delta + residual
       - 初始化时 delta ≈ 0，强制平滑先验 (z_t ≈ z_{t+1})
       - 网络只需学习小的增量，而不是绝对值
    6. 🔥 [FIX] 添加连续正弦时间编码 (支持不规则采样)
    
    使用方式 (在 FBI_SSM.run_backward_pass 中):
        smoother.precompute_history_features(Z_f, time_stamps)  # 只调用一次
        for t in reversed(range(T)):
            z_curr = smoother.forward_with_cache(t, z_next)  # 使用缓存
        smoother.clear_cache()  # 清理缓存
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, n_head=4, d_k=16, d_v=16, dropout=0.1, use_alibi=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim
        self.time_dim = time_dim
        self.use_alibi = use_alibi
        
        # ============================================================
        # 🔥 [FIX] 连续正弦时间编码 (支持不规则采样)
        # ============================================================
        self.time_embedding = ContinuousTimeEmbedding(d_model)
        
        # ============================================================
        # 1. 时序特征提取：双向 GRU (捕捉时序依赖)
        # ============================================================
        # 🔥 [FIX] GRU 输入增加 time embedding: keep_dim + d_model
        self.temporal_gru = nn.GRU(
            input_size=keep_dim + d_model,  # 🔥 增加 time embedding
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # ============================================================
        # 2. Cross Attention (用于从历史中提取相关信息)
        # ============================================================
        self.query_proj = nn.Linear(keep_dim, d_model)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        
        # ============================================================
        # 3. Local Attention (时序局部性归纳偏置)
        # ============================================================
        # 用 1D Conv 模拟 local window attention
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model // 4  # Grouped conv 更高效
        )
        
        # ============================================================
        # 4. 融合层
        # ============================================================
        # 输入: GRU_feat + Attention_feat + Local_feat + Z_next
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3 + keep_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        
        # ============================================================
        # 5. 输出层 (Delta + Residual)
        # 🔥 改进：预测 delta 而不是绝对值，加入 residual connection
        #    z_t = z_{t+1} + delta + residual
        #    这样初始化时 z_t ≈ z_{t+1}，强制平滑先验
        # ============================================================
        self.delta_head = nn.Linear(d_model, keep_dim)
        
        self.noise_fusion = nn.Sequential(
            nn.Linear(d_model + keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.residual_head = nn.Linear(d_model, keep_dim)
        
        # 🔥 Small-Init: 初始化为 0，使得初始输出 z_t = z_{t+1} (平滑先验)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.residual_head.bias, 0.0)
        
        # ============================================================
        # 🔥 缓存 (用于预计算)
        # ============================================================
        self._cache = None

    def precompute_history_features(self, H_full, time_stamps=None):
        """
        🔥 预计算 H 的所有特征，只调用一次！
        
        Args:
            H_full: [B, T, keep_dim] - 完整的 Filter 输出 Z_{1:T}
            time_stamps: [B, T] 或 [T,] 或 None - 连续时间戳
        
        存储到 self._cache:
            - gru_out: [B, T, d_model] - GRU 输出
            - H_proj: [B, T, d_model] - 投影后的 H (带 time embedding)
            - local_out: [B, d_model, T] - Local Conv 输出
            - alibi_bias: ALiBi 位置偏置
        """
        B, T, _ = H_full.shape
        device = H_full.device
        
        # 🔥 [FIX] 生成时间编码
        if time_stamps is None:
            time_stamps = torch.arange(T, dtype=torch.float32, device=device)
        
        if time_stamps.dim() == 1:
            # (T,) -> (T, d_model) -> (1, T, d_model) -> broadcast
            time_emb = self.time_embedding(time_stamps)  # (T, d_model)
            time_emb = time_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)
        else:
            # (B, T) -> (B, T, d_model)
            time_emb = self.time_embedding(time_stamps)
        
        # 🔥 [FIX] GRU 输入拼接 H 和 time embedding
        gru_input = torch.cat([H_full, time_emb], dim=-1)  # (B, T, keep_dim + d_model)
        
        # 1. GRU 特征 (一次性计算整个序列)
        gru_out, _ = self.temporal_gru(gru_input)  # (B, T, d_model)
        
        # 2. H 投影 (一次性计算)
        H_proj = self.query_proj(H_full)  # (B, T, d_model)
        
        # 🔥 [FIX] 给 H_proj 也加上时间编码
        H_proj = H_proj + time_emb  # (B, T, d_model)
        
        # 3. Local Conv (一次性计算)
        local_in = H_proj.permute(0, 2, 1)  # (B, d_model, T)
        local_out = self.local_conv(local_in)  # (B, d_model, T)
        
        # 🔥 [FIX] 预计算 ALiBi bias
        alibi_bias = None
        if self.use_alibi:
            from nets.salt_layers import get_alibi_bias
            alibi_bias = get_alibi_bias(T, self.nhead, device)
        
        # 存储缓存
        self._cache = {
            'gru_out': gru_out,
            'H_proj': H_proj,
            'local_out': local_out,
            'alibi_bias': alibi_bias,
            'T': T,
            'B': B,
        }
    
    def clear_cache(self):
        """清理缓存"""
        self._cache = None
    
    def forward_with_cache(self, t, z_next, noise=None):
        """
        🔥 使用预计算的缓存进行 forward，O(1) 复杂度！
        
        Args:
            t: int, 当前时间步 (0 到 T-2)
            z_next: [B, keep_dim] - Z_{t+1}
            noise: 可选的噪声
        
        Returns:
            z_curr: [B, keep_dim] - Z_t 的估计
        """
        assert self._cache is not None, "Must call precompute_history_features() first!"
        
        B = z_next.shape[0]
        device = z_next.device
        
        # 从缓存中取出预计算的特征
        gru_out = self._cache['gru_out']
        H_proj = self._cache['H_proj']
        local_out = self._cache['local_out']
        alibi_bias_full = self._cache['alibi_bias']
        
        # 取对应时间步的特征
        # 注意：我们需要 Z_{1:t+1} 的信息来推断 Z_t
        # 所以取 index 0 到 t (包含 t)，最后一个是 t
        
        # GRU 特征：取 t+1 位置 (因为 GRU 输出是双向的，包含了两边的信息)
        # 这里取 t 位置，因为我们要推断的是 Z_t
        gru_feat = gru_out[:, t, :]  # (B, d_model)
        
        # Cross Attention: 用 z_next 作为 query，H_{1:t+1} 作为 key/value
        z_next_emb = self.query_proj(z_next)  # (B, d_model)
        query = z_next_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # 只取 H_proj 的前 t+1 个时间步 (index 0 到 t)
        H_proj_slice = H_proj[:, :t+1, :]  # (B, t+1, d_model)
        
        # 🔥 [FIX] 添加 ALiBi bias 到 cross attention
        attn_mask = None
        if alibi_bias_full is not None and t+1 > 1:
            # alibi_bias_full: (nhead, T, T), 截取 query=1, key=t+1 的部分
            # 对于 cross-attention: query 长度 = 1, key 长度 = t+1
            # 我们需要 (nhead, 1, t+1) 的 bias
            # 但是标准 cross-attn 不直接支持 ALiBi，这里简化处理
            pass  # ALiBi 主要用于 self-attention，cross-attn 暂时不加
        
        attn_out, _ = self.cross_attn(query, H_proj_slice, H_proj_slice)
        attn_feat = attn_out.squeeze(1)  # (B, d_model)
        
        # Local 特征：取 t 位置
        local_feat = local_out[:, :, t]  # (B, d_model)
        
        # 融合
        combined = torch.cat([gru_feat, attn_feat, local_feat, z_next], dim=-1)
        fused = self.fusion(combined)  # (B, d_model)
        
        # 🔥 输出: Delta 预测 + Residual Connection
        # z_t = z_{t+1} + delta + residual
        z_delta = self.delta_head(fused)
        
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        noise_input = torch.cat([fused, noise], dim=-1)
        noise_feat = self.noise_fusion(noise_input)
        z_residual = self.residual_head(noise_feat)
        
        # 🔥 Residual connection: z_t = z_next + delta + residual
        z_curr = z_next + z_delta + z_residual
        
        return z_curr

    def forward(self, H, z_next, noise=None, time_stamps=None):
        """
        原始接口 (向后兼容)，每次调用都重新计算 H 的特征。
        
        ⚠️ 注意：如果在循环中调用，建议使用 precompute + forward_with_cache 代替！
        
        H: [B, t+1, keep_dim] - History (Z_1 到 Z_{t+1})
        z_next: [B, keep_dim] - Z_{t+1} (当前要往回推的点)
        time_stamps: [B, t+1] 或 [t+1,] 或 None - 时间戳 (可选)
        
        返回: Z_t 的估计
        """
        B = z_next.shape[0]
        seq_len = H.shape[1]
        device = z_next.device
        
        # 🔥 [FIX] 生成时间编码
        if time_stamps is None:
            time_stamps = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        if time_stamps.dim() == 1:
            time_emb = self.time_embedding(time_stamps)  # (seq_len, d_model)
            time_emb = time_emb.unsqueeze(0).expand(B, -1, -1)  # (B, seq_len, d_model)
        else:
            time_emb = self.time_embedding(time_stamps)  # (B, seq_len, d_model)
        
        # 🔥 [FIX] GRU 输入拼接 H 和 time embedding
        gru_input = torch.cat([H, time_emb], dim=-1)  # (B, seq_len, keep_dim + d_model)
        
        # ============================================================
        # A. 时序特征提取 (GRU)
        # ============================================================
        gru_out, _ = self.temporal_gru(gru_input)  # (B, t+1, d_model)
        gru_feat = gru_out[:, -1, :]  # 取最后一个时间步 (B, d_model)
        
        # ============================================================
        # B. Cross Attention (全局依赖)
        # ============================================================
        z_next_emb = self.query_proj(z_next)  # (B, d_model)
        query = z_next_emb.unsqueeze(1)  # (B, 1, d_model)
        
        H_proj = self.query_proj(H)  # (B, t+1, d_model)
        # 🔥 [FIX] 给 H_proj 也加上时间编码
        H_proj = H_proj + time_emb  # (B, t+1, d_model)
        
        attn_out, _ = self.cross_attn(query, H_proj, H_proj)
        attn_feat = attn_out.squeeze(1)  # (B, d_model)
        
        # ============================================================
        # C. Local Attention (局部时序依赖)
        # ============================================================
        # (B, t+1, d_model) -> (B, d_model, t+1) -> Conv -> (B, d_model, t+1)
        local_in = H_proj.permute(0, 2, 1)
        local_out = self.local_conv(local_in)
        local_feat = local_out[:, :, -1]  # 取最后位置 (B, d_model)
        
        # ============================================================
        # D. 融合所有特征
        # ============================================================
        combined = torch.cat([gru_feat, attn_feat, local_feat, z_next], dim=-1)
        fused = self.fusion(combined)  # (B, d_model)
        
        # ============================================================
        # E. 输出 (Delta + Residual)
        # 🔥 z_t = z_next + delta + residual (平滑先验)
        # ============================================================
        z_delta = self.delta_head(fused)
        
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        noise_input = torch.cat([fused, noise], dim=-1)
        noise_feat = self.noise_fusion(noise_input)
        z_residual = self.residual_head(noise_feat)
        
        # 🔥 Residual connection: z_t = z_next + delta + residual
        z_curr = z_next + z_delta + z_residual
        
        return z_curr


# ============================================================
# 🔥🔥 SSM_Smoother_V3: 符合理论的 Backward Pass 实现
# ============================================================
# 
# 理论公式 (Theorem 2 - Backward imputation recursion):
#   Z̃_t = q(Z̃_{t+1}, X_{1:t}^o, ξ_t)
#
# 关键改进:
# 1. 输入是原始观测 X_{1:t}^o（不是 encoder 输出 H）
# 2. 使用连续正弦时间编码（支持不规则采样）
# 3. 正确的信息流：只使用到 t 为止的观测
# 4. 结合 Z_{t+1} 和 X_{1:t}^o 的信息
# ============================================================

class SSM_Smoother_V3(nn.Module):
    """
    符合理论的 Smoother 实现 (Backward Pass)
    
    理论公式：
        Z̃_t = q(Z̃_{t+1}, X_{1:t}^o, ξ_t)
    
    输入：
        - Z̃_{t+1}: 下一个时间步的平滑后状态
        - X_{1:t}^o: 从时间 1 到 t 的原始观测（带 mask）
        - ξ_t: 噪声
    
    关键改进：
        1. 使用原始观测 X^o 而不是 encoder 输出 H
        2. 使用连续正弦时间编码（支持不规则采样）
        3. Cross attention 带 causal mask（只看 1:t 的观测）
        4. 时间编码帮助模型理解"越近的观测越重要"
    """
    def __init__(self, feature_dim, keep_dim, time_dim, d_model=64, 
                 n_head=4, d_k=32, d_v=32, dropout=0.1, k_order=1, use_alibi=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.k_order = k_order
        self.use_alibi = use_alibi
        
        # ============================================================
        # 1. 观测数据编码 (X^o + mask -> d_model)
        # ============================================================
        # 输入: [x, mask] (2 channels)
        self.obs_projection = nn.Conv1d(2, d_model, 1)
        nn.init.kaiming_normal_(self.obs_projection.weight)
        
        # 🔥 连续正弦时间编码 (支持不规则采样)
        self.time_embedding = ContinuousTimeEmbedding(d_model)
        
        # 特征编码 (每个特征有自己的"身份证")
        self.feature_embedding = nn.Embedding(feature_dim, d_model)
        
        # ============================================================
        # 2. 观测序列 Transformer (处理 X_{1:t}^o)
        # ============================================================
        # 时间方向 attention（带 causal mask）
        self.time_transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, 
            dim_feedforward=d_model * 2, 
            activation="gelu", batch_first=True, dropout=dropout
        )
        
        # 特征方向 attention（同一时间点的特征可以互相看）
        self.feature_transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, dropout=dropout
        )
        
        # ============================================================
        # 3. Z_{t+1} 投影
        # ============================================================
        self.z_next_proj = nn.Linear(keep_dim, d_model)
        
        # ============================================================
        # 4. Cross Attention: Z_{t+1} attend to X_{1:t}^o
        # ============================================================
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        
        # ============================================================
        # 5. Future Window Aggregation (for k > 1)
        # ============================================================
        if k_order > 1:
            self.future_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
            self.future_proj = nn.Linear(keep_dim, d_model)
            self.future_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.future_out_proj = nn.Linear(d_model, keep_dim)
            nn.init.xavier_uniform_(self.future_query)
        
        # ============================================================
        # 6. Fusion Layer
        # ============================================================
        # 输入: cross_attn_feat + z_next_emb + aggregated_obs
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        
        # ============================================================
        # 7. Output Layer (Delta + Residual)
        # ============================================================
        self.delta_head = nn.Linear(d_model, keep_dim)
        
        self.noise_fusion = nn.Sequential(
            nn.Linear(d_model + keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.residual_head = nn.Linear(d_model, keep_dim)
        
        # Small-Init for smooth prior
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.residual_head.bias, 0.0)
        
        # Cache
        self._cache = None
        self._causal_mask_cache = {}
    
    def _get_causal_mask(self, L, device):
        """生成 causal mask (只能看当前和过去)"""
        cache_key = (L, device)
        if cache_key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]
    
    def precompute_observation_features(self, X_obs, mask_obs, time_stamps=None):
        """
        预计算观测数据的特征（只调用一次）
        
        Args:
            X_obs: (B, T, p) 原始观测数据
            mask_obs: (B, T, p) 观测 mask
            time_stamps: (B, T) 或 (T,) 或 None 连续时间戳
        """
        B, T, p = X_obs.shape
        device = X_obs.device
        
        # ============================================================
        # Step 1: 输入处理
        # ============================================================
        mask_float = mask_obs.float()
        x_masked = X_obs * mask_float  # 缺失位置填 0
        
        # [x, mask] -> (B, T, p, 2) -> (B, 2, p, T)
        x_input = torch.stack([x_masked, mask_float], dim=-1)
        x_input = x_input.permute(0, 3, 2, 1)  # (B, 2, p, T)
        
        # 投影: (B, 2, p*T) -> (B, d_model, p*T)
        x_flat = x_input.reshape(B, 2, p * T)
        x_proj = self.obs_projection(x_flat)
        x_proj = F.relu(x_proj)
        x_proj = x_proj.reshape(B, self.d_model, p, T)  # (B, d_model, p, T)
        
        # ============================================================
        # Step 2: 添加 Embedding (Feature + Time)
        # ============================================================
        # Feature embedding: (p,) -> (p, d_model) -> (1, d_model, p, 1)
        feat_emb = self.feature_embedding(torch.arange(p, device=device))
        feat_emb = feat_emb.T.unsqueeze(0).unsqueeze(-1)  # (1, d_model, p, 1)
        
        # 🔥 Time embedding (连续正弦编码)
        if time_stamps is None:
            time_stamps = torch.arange(T, dtype=torch.float32, device=device)
        
        if time_stamps.dim() == 1:
            time_emb = self.time_embedding(time_stamps)  # (T, d_model)
            time_emb = time_emb.T.unsqueeze(0).unsqueeze(2)  # (1, d_model, 1, T)
        else:
            time_emb = self.time_embedding(time_stamps)  # (B, T, d_model)
            time_emb = time_emb.permute(0, 2, 1).unsqueeze(2)  # (B, d_model, 1, T)
        
        # 叠加 embedding
        x_proj = x_proj + feat_emb + time_emb  # (B, d_model, p, T)
        
        # ============================================================
        # Step 3: 时间方向 Transformer (带 causal mask)
        # ============================================================
        # (B, d_model, p, T) -> (B*p, T, d_model)
        x_time = x_proj.permute(0, 2, 3, 1).reshape(B * p, T, self.d_model)
        
        causal_mask = self._get_causal_mask(T, device)
        x_time = self.time_transformer(x_time, src_mask=causal_mask)
        
        x_time = x_time.reshape(B, p, T, self.d_model).permute(0, 3, 1, 2)  # (B, d_model, p, T)
        
        # ============================================================
        # Step 4: 特征方向 Transformer (无 mask)
        # ============================================================
        # (B, d_model, p, T) -> (B*T, p, d_model)
        x_feat = x_time.permute(0, 3, 2, 1).reshape(B * T, p, self.d_model)
        x_feat = self.feature_transformer(x_feat)
        x_feat = x_feat.reshape(B, T, p, self.d_model).permute(0, 3, 2, 1)  # (B, d_model, p, T)
        
        # ============================================================
        # Step 5: 聚合特征方向 (mean pooling over features)
        # ============================================================
        # (B, d_model, p, T) -> (B, d_model, T) -> (B, T, d_model)
        obs_features = x_feat.mean(dim=2).permute(0, 2, 1)  # (B, T, d_model)
        
        # ============================================================
        # 缓存
        # ============================================================
        alibi_bias = None
        if self.use_alibi:
            from nets.salt_layers import get_alibi_bias
            alibi_bias = get_alibi_bias(T, self.nhead, device)
        
        self._cache = {
            'obs_features': obs_features,  # (B, T, d_model)
            'alibi_bias': alibi_bias,
            'T': T,
            'B': B,
        }
    
    def clear_cache(self):
        self._cache = None
    
    def _aggregate_future_window(self, z_future_window):
        """聚合 k 个未来状态为单个表示"""
        B = z_future_window.shape[0]
        device = z_future_window.device
        
        if self.k_order == 1 or z_future_window.shape[1] == 1:
            return z_future_window[:, 0, :]
        
        future_proj = self.future_proj(z_future_window)
        query = self.future_query.expand(B, -1, -1)
        
        alibi_bias = None
        if self.use_alibi:
            from nets.salt_layers import get_alibi_bias
            k = z_future_window.shape[1]
            alibi_bias = get_alibi_bias(k, self.nhead, device)
            alibi_bias = alibi_bias[:, :1, :].unsqueeze(0)
        
        attn_out, _ = self.future_attn(query, future_proj, future_proj, alibi_bias=alibi_bias)
        return self.future_out_proj(attn_out.squeeze(1))
    
    def forward_with_cache(self, t, z_future_window, noise=None):
        """
        使用预计算缓存的 forward
        
        理论公式：Z̃_t = q(Z̃_{t+1}, X_{1:t}^o, ξ_t)
        
        Args:
            t: int, 当前时间步
            z_future_window: [B, k, keep_dim] 或 [B, keep_dim]
            noise: 可选噪声
        
        Returns:
            z_curr: [B, keep_dim]
        """
        assert self._cache is not None, "Must call precompute_observation_features() first!"
        
        device = z_future_window.device
        
        if z_future_window.dim() == 2:
            z_future_window = z_future_window.unsqueeze(1)
        
        B = z_future_window.shape[0]
        
        # 🔥 物理上最近的点用于残差连接
        z_next_physical = z_future_window[:, 0, :]
        
        # 聚合未来窗口用于 attention
        if self.k_order > 1:
            z_next_context = self._aggregate_future_window(z_future_window)
        else:
            z_next_context = z_next_physical
        
        # ============================================================
        # 从缓存获取观测特征
        # ============================================================
        obs_features = self._cache['obs_features']  # (B, T, d_model)
        alibi_bias_full = self._cache['alibi_bias']
        
        # 🔥 关键：只使用 X_{1:t}^o 的信息（到 t 为止）
        obs_features_1_to_t = obs_features[:, :t+1, :]  # (B, t+1, d_model)
        
        # ============================================================
        # Cross Attention: Z_{t+1} attend to X_{1:t}^o
        # ============================================================
        z_next_emb = self.z_next_proj(z_next_context)  # (B, d_model)
        query = z_next_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # ALiBi bias (slice for current t)
        alibi_bias = None
        if alibi_bias_full is not None:
            alibi_bias = alibi_bias_full[:, t:t+1, :t+1].unsqueeze(0)
        
        attn_out, _ = self.cross_attn(query, obs_features_1_to_t, obs_features_1_to_t, alibi_bias=alibi_bias)
        cross_feat = attn_out.squeeze(1)  # (B, d_model)
        
        # ============================================================
        # 聚合当前时间步的观测特征
        # ============================================================
        obs_feat_t = obs_features[:, t, :]  # (B, d_model)
        
        # ============================================================
        # Fusion
        # ============================================================
        combined = torch.cat([cross_feat, z_next_emb, obs_feat_t], dim=-1)
        fused = self.fusion(combined)
        
        # ============================================================
        # Output (Delta + Residual)
        # ============================================================
        z_delta = self.delta_head(fused)
        
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        noise_input = torch.cat([fused, noise], dim=-1)
        noise_feat = self.noise_fusion(noise_input)
        z_residual = self.residual_head(noise_feat)
        
        # 🔥 残差连接基于物理上最近的 Z_{t+1}
        z_curr = z_next_physical + z_delta + z_residual
        
        return z_curr


class SSM_Smoother_V2_KOrder(nn.Module):
    """
    K-order Markovian Smoother with ALiBi positional encoding.
    
    改进点:
    1. 接受 k 个未来状态 Z_{t+1:t+k} 而不是单个 z_{t+1}
    2. 使用 attention 聚合未来窗口
    3. 对 history attention 和 future attention 都应用 ALiBi bias
    4. Delta 预测 + residual connection
    
    对于 k=1，行为等价于 SSM_Smoother_V2。
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, 
                 n_head=4, d_k=16, d_v=16, dropout=0.1, k_order=1, use_alibi=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim
        self.time_dim = time_dim
        self.k_order = k_order
        self.use_alibi = use_alibi
        
        # ============================================================
        # 1. Future Window Aggregation (for k > 1)
        # ============================================================
        if k_order > 1:
            self.future_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
            self.future_proj = nn.Linear(keep_dim, d_model)
            self.future_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.future_out_proj = nn.Linear(d_model, keep_dim)
            nn.init.xavier_uniform_(self.future_query)
        
        # ============================================================
        # 2. Temporal Feature Extraction: Bidirectional GRU
        # ============================================================
        self.temporal_gru = nn.GRU(
            input_size=keep_dim,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # ============================================================
        # 3. Cross Attention (with ALiBi support)
        # ============================================================
        self.query_proj = nn.Linear(keep_dim, d_model)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        
        # ============================================================
        # 4. Local Attention (temporal locality)
        # ============================================================
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=max(1, d_model // 4)
        )
        
        # ============================================================
        # 5. Fusion Layer
        # ============================================================
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3 + keep_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        
        # ============================================================
        # 6. Output Layer (Delta + Residual)
        # ============================================================
        self.delta_head = nn.Linear(d_model, keep_dim)
        
        self.noise_fusion = nn.Sequential(
            nn.Linear(d_model + keep_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.residual_head = nn.Linear(d_model, keep_dim)
        
        # Small-Init for smooth prior
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.residual_head.bias, 0.0)
        
        # Cache for precomputed features
        self._cache = None

    def _aggregate_future_window(self, z_future_window):
        """
        聚合 k 个未来状态为单个表示。
        
        Args:
            z_future_window: [B, k, keep_dim] - Z_{t+1}...Z_{t+k}
        
        Returns:
            z_next_agg: [B, keep_dim] - 聚合后的未来状态
        """
        B = z_future_window.shape[0]
        device = z_future_window.device
        
        if self.k_order == 1 or z_future_window.shape[1] == 1:
            # 直接返回单个未来状态
            return z_future_window[:, 0, :]
        
        # Project future states
        future_proj = self.future_proj(z_future_window)  # [B, k, d_model]
        
        # Use learnable query to attend over future window
        query = self.future_query.expand(B, -1, -1)  # [B, 1, d_model]
        
        # Optional: Add ALiBi bias for future attention
        alibi_bias = None
        if self.use_alibi:
            from nets.salt_layers import get_alibi_bias
            k = z_future_window.shape[1]
            alibi_bias = get_alibi_bias(k, self.nhead, device)
            # Reshape for [B, n_head, 1, k]
            alibi_bias = alibi_bias[:, :1, :].unsqueeze(0)  # [1, n_head, 1, k]
        
        attn_out, _ = self.future_attn(query, future_proj, future_proj, alibi_bias=alibi_bias)
        z_next_agg = self.future_out_proj(attn_out.squeeze(1))  # [B, keep_dim]
        
        return z_next_agg

    def precompute_history_features(self, H_full):
        """
        预计算 history 特征，只调用一次。
        """
        B, T, _ = H_full.shape
        device = H_full.device
        
        # 1. GRU features
        gru_out, _ = self.temporal_gru(H_full)  # (B, T, d_model)
        
        # 2. H projection
        H_proj = self.query_proj(H_full)  # (B, T, d_model)
        
        # 3. Local Conv
        local_in = H_proj.permute(0, 2, 1)  # (B, d_model, T)
        local_out = self.local_conv(local_in)  # (B, d_model, T)
        
        # 4. Precompute ALiBi bias for history attention
        alibi_bias = None
        if self.use_alibi:
            from nets.salt_layers import get_alibi_bias
            alibi_bias = get_alibi_bias(T, self.nhead, device)  # [n_head, T, T]
        
        self._cache = {
            'gru_out': gru_out,
            'H_proj': H_proj,
            'local_out': local_out,
            'alibi_bias': alibi_bias,
            'T': T,
            'B': B,
        }
    
    def clear_cache(self):
        self._cache = None

    def forward_with_cache(self, t, z_future_window, noise=None):
        """
        使用预计算缓存的 forward (k-order 版本).
        
        Args:
            t: int, 当前时间步 (0 到 T-k-1)
            z_future_window: [B, k, keep_dim] for k>1, or [B, keep_dim] for k=1
            noise: optional noise
        
        Returns:
            z_curr: [B, keep_dim] - estimated Z_t
        """
        assert self._cache is not None, "Must call precompute_history_features() first!"
        
        device = z_future_window.device
        
        # Handle different input shapes
        if z_future_window.dim() == 2:
            z_future_window = z_future_window.unsqueeze(1)  # [B, 1, keep_dim]
        
        B = z_future_window.shape[0]
        
        # 🔥 [FIX] 物理上最近的点永远是 Z_{t+1}，用于残差连接
        # 这保证了轨迹连续性：Z_t 最应该接近的就是 Z_{t+1}
        z_next_physical = z_future_window[:, 0, :]  # [B, keep_dim] - 永远是 Z_{t+1}
        
        # Aggregate future window (for k>1, uses attention; for k=1, just returns first)
        # z_next_context: 用于 attention 的上下文信息，融合了 k 步的未来趋势
        if self.k_order > 1:
            z_next_context = self._aggregate_future_window(z_future_window)
        else:
            z_next_context = z_next_physical
        
        # Get cached features
        gru_out = self._cache['gru_out']
        H_proj = self._cache['H_proj']
        local_out = self._cache['local_out']
        alibi_bias_full = self._cache['alibi_bias']
        
        # GRU feature at position t
        gru_feat = gru_out[:, t, :]  # (B, d_model)
        
        # Cross Attention with ALiBi
        # 🔥 [FIX] 使用 z_next_context（聚合的未来趋势）作为 attention query
        # 这让模型能看到更远的未来信息，提高长程依赖建模能力
        z_next_emb = self.query_proj(z_next_context)  # (B, d_model)
        query = z_next_emb.unsqueeze(1)  # (B, 1, d_model)
        H_proj_slice = H_proj[:, :t+1, :]  # (B, t+1, d_model)
        
        # Get ALiBi bias slice for this attention
        alibi_bias = None
        if alibi_bias_full is not None:
            # Shape: [n_head, T, T] -> [n_head, 1, t+1]
            alibi_bias = alibi_bias_full[:, t:t+1, :t+1].unsqueeze(0)  # [1, n_head, 1, t+1]
        
        attn_out, _ = self.cross_attn(query, H_proj_slice, H_proj_slice, alibi_bias=alibi_bias)
        attn_feat = attn_out.squeeze(1)  # (B, d_model)
        
        # Local feature at position t
        local_feat = local_out[:, :, t]  # (B, d_model)
        
        # Fusion
        # 🔥 [FIX] 使用 z_next_context 作为 fusion 输入（包含未来趋势信息）
        combined = torch.cat([gru_feat, attn_feat, local_feat, z_next_context], dim=-1)
        fused = self.fusion(combined)  # (B, d_model)
        
        # Delta prediction
        z_delta = self.delta_head(fused)
        
        # Residual with noise
        if noise is None:
            noise = torch.randn(B, self.keep_dim, device=device)
        
        noise_input = torch.cat([fused, noise], dim=-1)
        noise_feat = self.noise_fusion(noise_input)
        z_residual = self.residual_head(noise_feat)
        
        # 🔥 [FIX] Final: z_t = z_next_physical + delta + residual
        # 关键修正：残差连接基于物理上最近的 Z_{t+1}，而不是聚合后的值
        # 这保证了轨迹连续性：模型只需要学习 Z_t 相对于 Z_{t+1} 的增量
        # delta 和 residual 已经通过 z_next_context 学习到了未来趋势
        z_curr = z_next_physical + z_delta + z_residual
        
        return z_curr


# ============================================================
# 🔥🔥 SSM_Smoother_V4: Cross Attention Smoother (Non-Markovian)
# ============================================================
# 
# 理论公式：
#   Z̃_t = q(Z̃_{t+1}, Z^f_{1:t}, ξ_t)
#
# 与 V2/V3 的区别：
# - V2: 使用 encoder 输出 H (= Z_f) + GRU 等时序网络
# - V3: 使用原始观测 X^o
# - V4: 纯 Cross Attention，Query=Z_{t+1}, Key/Value=Z_f_{1:t}
#
# 优点：
# 1. 架构更简洁，只有 cross attention
# 2. 使用 cache 加速计算，避免重复计算 K/V
# 3. 更符合理论：平滑器只依赖于 filter 输出和未来平滑状态
# ============================================================

class SSM_Smoother_V4(nn.Module):
    """
    SSM Smoother - V4: Cross Attention (Non-Markovian)
    
    理论公式：
        Z̃_t = q(Z̃_{t+1}, Z^f_{1:t})
        
    其中：
        - Z̃_{t+1}: 下一个时间步的平滑后状态
        - Z^f_{1:t}: 从时间 1 到 t 的 filter 输出
    
    实现：
        Query = Z̃_{t+1}
        Key/Value = Z^f_{1:t}
        Output = Z̃_t
        
    注意（代码下标 vs 数学下标）：
        - 代码下标 t=0 对应数学下标 t=1
        - 代码中 H[:, :t+1, :] 对应数学中 Z^f_{1:t}
        - 当 t=0 时（数学 t=1），Z̃_1 = q(Z̃_2, Z^f_1)
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, 
                 n_head=4, d_k=32, d_v=32, dropout=0.1, d_inner=256):
        super().__init__()
        self.d_model = d_model
        self.keep_dim = keep_dim 
        self.time_dim = time_dim
        self.nhead = n_head
        
        if d_inner is None:
            d_inner = d_model * 2
        
        # ============================================================
        # 1. Projection Layers
        # ============================================================
        # Query: Z_{t+1} -> d_model
        self.query_proj = nn.Linear(keep_dim, d_model)
        
        # Key/Value: Z_f (history) -> d_model
        self.key_proj = nn.Linear(keep_dim, d_model)
        self.value_proj = nn.Linear(keep_dim, d_model)

        # ============================================================
        # 2. Cross Attention Mechanism
        # ============================================================
        self.cross_attn = RoPEMultiHeadAttention(
            n_head=n_head, 
            d_model=d_model, 
            d_k=d_k, 
            d_v=d_v, 
            attn_dropout=dropout,
            max_len=50 # 传入 max_len
        )
        
        # ============================================================
        # 3. Output Processing
        # ============================================================
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Position-wise FFN
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        
        # ============================================================
        # 4. Output Projection (Delta + Residual)
        # ============================================================
        # 预测 delta 而不是绝对值，强制平滑先验
        self.delta_head = nn.Linear(d_model, keep_dim)
        
        # Noise fusion for stochastic residual
        # self.noise_fusion = nn.Sequential(
        #     nn.Linear(d_model + keep_dim, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, d_model),
        #     nn.GELU()
        # )
        # self.residual_head = nn.Linear(d_model, keep_dim)
        
        # 🔥 Small-Init: 初始化为接近 0，使得初始输出 z_t ≈ z_{t+1} (平滑先验)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)
        # nn.init.normal_(self.residual_head.weight, mean=0.0, std=0.01)
        # nn.init.constant_(self.residual_head.bias, 0.0)

        # 🔥 Cache for efficient backward pass
        self._cache = {}

    def precompute_history_features(self, H_full, time_stamps=None):
        """
        在 Backward Pass 开始前，一次性预计算所有历史步的 Key 和 Value。
        
        Args:
            H_full: [B, T, keep_dim] - Forward Pass 的输出 Z_f (filter output)
            time_stamps: 时间戳（本版本暂不使用，保留接口兼容性）
        
        调用时机：在 backward loop 开始前调用一次
        """
        B, T, d = H_full.shape
        device = H_full.device
        
        # 预计算 Key 和 Value: [B, T, d_model]
        key_emb_full = self.key_proj(H_full)
        value_emb_full = self.value_proj(H_full)
        
        self._cache = {
            'key_emb_full': key_emb_full,      # [B, T, d_model]
            'value_emb_full': value_emb_full,  # [B, T, d_model]
            'H_full': H_full,                  # [B, T, keep_dim] 保存原始 H 用于 t=0 的特殊处理
            'B': B,
            'T': T
        }

    def forward_with_cache(self, t, z_next, noise=None):
        """
        使用预计算缓存的 forward（用于 Backward Loop）
        
        理论公式：Z̃_{t+1} = q(Z̃_{t+2}, Z^f_{1:t+1})
        
        其中 t 是代码下标 (0-indexed)：
        - 代码下标 t 对应数学位置 t+1
        - 我们要输出的是 Z̃[t]（代码）= Z̃_{t+1}（数学）
        - 需要 Z^f_{1:t+1}（数学），即 H[:, :t+1, :]（代码），共 t+1 个元素
        - z_next = Z̃[t+1]（代码）= Z̃_{t+2}（数学）
        
        Args:
            t: int, 当前时间步索引（代码下标，0-indexed）
               - t=0: 输出 Z̃_1（数学），需要 Z^f_1（1 个元素）
               - t=T-2: 输出 Z̃_{T-1}（数学），需要 Z^f_{1:T-1}（T-1 个元素）
            z_next: [B, keep_dim] - Z̃_{t+2}（数学下标）= Z̃[t+1]（代码下标）
            noise: 可选的噪声输入
        
        Returns:
            z_curr: [B, keep_dim] - Z̃_{t+1}（数学下标）= Z̃[t]（代码下标）
        """
        assert self._cache, "Must call precompute_history_features first!"
        
        device = z_next.device
        B = z_next.shape[0]
        
        # ============================================================
        # Step 1: 准备 Query [B, 1, d_model]
        # ============================================================
        query = self.query_proj(z_next).unsqueeze(1)  # [B, 1, d_model]
        
        # ============================================================
        # Step 2: 从 Cache 取出历史 Key/Value
        # ============================================================
        # 理论公式：Z̃_{t+1} = q(Z̃_{t+2}, Z^f_{1:t+1})
        # - t 是代码下标 (0-indexed)
        # - 我们要计算的是 Z̃_{t+1}（数学下标）
        # - 需要 Z^f_{1:t+1}（数学下标，共 t+1 个元素）
        # - 代码中 H[:, :t+1, :] 取代码下标 0 到 t，对应数学下标 1 到 t+1
        k_hist = self._cache['key_emb_full'][:, :t+1, :]    # [B, t+1, d_model]
        v_hist = self._cache['value_emb_full'][:, :t+1, :]  # [B, t+1, d_model]
        
        # ============================================================
        # Step 3: Cross Attention
        # ============================================================
        # Query: [B, 1, d_model], Key/Value: [B, t+1, d_model]
        # Output: [B, 1, d_model]
        # RoPE position: query 位于 t+1（数学下标），key 从 0 开始到 t
        context, _ = self.cross_attn(
            q = query,
            k = k_hist,
            v = v_hist, 
            attn_mask = None,
            alibi_bias = None,
            start_pos_q = t + 1,  # query 是 Z̃_{t+1}，位于数学位置 t+1
            start_pos_k = 0       # key 从 Z^f_1 开始，位置从 0 开始
        )
        
        # ============================================================
        # Step 4: Residual Connection & Layer Norm & FFN
        # ============================================================
        # Residual in embedding space
        context = context + query  # [B, 1, d_model]
        context = self.layer_norm(context)
        
        # Position-wise FFN
        context = self.pos_ffn(context)  # [B, 1, d_model]
        
        fused = context.squeeze(1)  # [B, d_model]
        
        # ============================================================
        # Step 5: Output (Delta + Residual)
        # ============================================================
        # Delta prediction
        z_delta = self.delta_head(fused)  # [B, keep_dim]
        
        # # Noise-based residual
        # if noise is None:
        #     noise = torch.randn(B, self.keep_dim, device=device)
        
        # noise_input = torch.cat([fused, noise], dim=-1)
        # noise_feat = self.noise_fusion(noise_input)
        # z_residual = self.residual_head(noise_feat)  # [B, keep_dim]
        
        # 🔥 Final: z_t = z_next + delta + residual (平滑先验)
        # z_curr = z_next + z_delta + z_residual
        z_curr = z_next + z_delta 
        
        return z_curr
    
    def forward(self, H, z_next, noise=None):
        """
        原始接口（向后兼容）- 不使用 cache，每次重新计算
        
        理论公式：Z̃_{seq_len} = q(Z̃_{seq_len+1}, Z^f_{1:seq_len})
        
        其中 seq_len = H.shape[1]（历史序列长度）
        
        ⚠️ 与 forward_with_cache 的对应关系：
        - forward_with_cache(t, z_next) ≡ forward(H[:, :t+1, :], z_next)
        - 历史序列长度 = 输出的数学下标
        
        Args:
            H: [B, seq_len, keep_dim] - filter output Z^f_{1:seq_len}
               序列长度 seq_len 对应数学下标 1 到 seq_len
            z_next: [B, keep_dim] - Z̃_{seq_len+1}（下一个时间步的平滑状态）
            noise: 可选噪声
            
        Returns:
            z_curr: [B, keep_dim] - Z̃_{seq_len}（当前时间步的平滑状态）
        """
        B = z_next.shape[0]
        seq_len = H.shape[1]  # 序列长度 = seq_len（数学下标 1:seq_len）
        device = z_next.device
        
        # ============================================================
        # Step 1: 准备 Query [B, 1, d_model]
        # ============================================================
        # Query = Z̃_{seq_len+1}（下一个时间步的平滑状态）
        query = self.query_proj(z_next).unsqueeze(1)  # [B, 1, d_model]
        
        # ============================================================
        # Step 2: 准备 Key/Value [B, seq_len, d_model]
        # ============================================================
        # Key/Value = Z^f_{1:seq_len}（从时间 1 到 seq_len 的 filter 输出序列）
        # H 已经是 [B, seq_len, keep_dim]，直接投影即可
        k = self.key_proj(H)   # [B, seq_len, d_model]
        v = self.value_proj(H)  # [B, seq_len, d_model]
        
        # ============================================================
        # Step 3: Cross Attention
        # ============================================================
        # Query: [B, 1, d_model] at position seq_len (RoPE)
        # Key/Value: [B, seq_len, d_model] at positions 0 to seq_len-1
        context, _ = self.cross_attn(
            q = query,
            k = k,
            v = v, 
            attn_mask = None,
            alibi_bias = None,
            start_pos_q = seq_len,  # query 是 Z̃_{seq_len+1}，RoPE 位置 = seq_len
            start_pos_k = 0         # key 从 Z^f_1 开始，位置从 0 开始
        )
        
        # ============================================================
        # Step 4: Residual + LayerNorm + FFN
        # ============================================================
        context = context + query  # [B, 1, d_model]
        context = self.layer_norm(context)
        context = self.pos_ffn(context)
        fused = context.squeeze(1)  # [B, d_model]
        
        # ============================================================
        # Step 5: Output (Delta + Residual)
        # ============================================================
        z_delta = self.delta_head(fused)  # [B, keep_dim]
        
        # if noise is None:
        #     noise = torch.randn(B, self.keep_dim, device=device)
        
        # noise_input = torch.cat([fused, noise], dim=-1)
        # noise_feat = self.noise_fusion(noise_input)
        # z_residual = self.residual_head(noise_feat)  # [B, keep_dim]
        
        # 🔥 Final: z_t = z_next + delta + residual (平滑先验)
        # z_curr = z_next + z_delta + z_residual
        z_curr = z_next + z_delta 
        
        return z_curr

    def clear_cache(self):
        """清理缓存"""
        self._cache = {}


######################## --- v5
class SSM_Smoother_V5(nn.Module):
    """
    SSM Smoother - V1 版本 (residual MLP)
    
    公式: z_t^s = z_t^f + residual_mlp(z_t^f, z_{t+1}^s, noise)
    
    这是最简单的版本，直接学习残差,并且不加noise
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, n_head=4, d_k=16, d_v=16, dropout=0.1, transition_g=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = n_head
        self.keep_dim = keep_dim 
        self.time_dim = time_dim
        
        # ============================================================
        # V1 版本: Residual MLP
        # ============================================================
        self.residual_mlp = nn.Sequential(
            nn.Linear(keep_dim * 2, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, keep_dim)
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        
        # 🔥 Cache (保留用于兼容)
        self._cache = {}
    # ============================================================
    # V1 forward: z_s_t = z_f_t + residual
    # ============================================================
    def forward(self, z_f_t, z_s_next, noise=None, noise_scale=0.1):
        """
        V1 版本: z_t^s = z_t^f + residual_mlp(z_t^f, z_{t+1}^s, scaled_noise)
        
        Args:
            z_f_t: [B, keep_dim] - 当前时刻的 filter 输出 z_t^f
            z_s_next: [B, keep_dim] - 下一时刻的 smoothed z_{t+1}^s
            noise: [B, keep_dim] - 随机噪声 (用于生成模型)
        
        Returns:
            z_s_t: [B, keep_dim] - 当前时刻的 smoothed z_t^s
        """
        B = z_f_t.shape[0]
        device = z_f_t.device
        
        # if noise is None:
            # noise = torch.randn(B, self.keep_dim, device=device)
        
        # scaled_noise = noise * noise_scale
        fusion_input = torch.cat([z_f_t, z_s_next], dim=-1)
        z_residual = self.residual_mlp(fusion_input)
        z_s_t = z_f_t + z_residual
        
        # 数值稳定性
        z_s_t = torch.clamp(z_s_t, min=-100.0, max=100.0)
        return z_s_t

    def precompute_history_features(self, H_full, time_stamps=None):
        """
        🔥 新版不需要 precompute，但保留接口兼容性
        只需存储 H_full 供 backward loop 使用
        """
        self._cache = {'H_full': H_full}
    
    def clear_cache(self):
        """清理 cache"""
        self._cache = {}


######################## --- v6
class SSM_Smoother_V6(nn.Module):
    """
    SSM Smoother - V6: Backward at any time t (Non-Markovian)
    
    理论公式：
        Ž_s = BackwardFunc(Ž_{s+1:t}, z_s^f, ...)
        
    关键改进（相对于 V4）：
        1. Conditioning Input: 使用 forward encoder feature z_s^f 作为 Query（而非 Ž_{s+1}）
        2. Extended Dependency: Key/Value 来自所有已生成的 backward states Ž_{s+1:t}
        
    实现：
        Query = z_s^f (当前时刻的 forward feature)
        Key/Value = Ž_{s+1:t} (所有已生成的 backward states)
        Output = Ž_s
        
    Backward Loop 过程 (从 t-1 到 0):
        - 初始化: Ž_t = z_t^f (boundary condition)
        - s = t-1: Q = z_{t-1}^f, K/V = [Ž_t] (1 个元素)
        - s = t-2: Q = z_{t-2}^f, K/V = [Ž_{t-1}, Ž_t] (2 个元素)
        - ...
        - s = 0:   Q = z_0^f, K/V = [Ž_1, ..., Ž_t] (t 个元素)
        
    注意：
        - backward states 按时间顺序存储（不是生成顺序）
        - 生成 Ž_s 时，attend to Ž_{s+1:t}（时间上在 s 之后的所有状态）
    """
    def __init__(self, feature_dim=4, keep_dim=2, time_dim=48, d_model=64, 
                 n_head=4, d_k=32, d_v=32, dropout=0.1, d_inner=None):
        super().__init__()
        self.d_model = d_model
        self.keep_dim = keep_dim 
        self.time_dim = time_dim
        self.nhead = n_head
        
        if d_inner is None:
            d_inner = d_model * 2
        
        # ============================================================
        # 1. Projection Layers
        # ============================================================
        # Query: z_s^f (forward feature) -> d_model
        self.query_proj = nn.Linear(keep_dim, d_model)
        
        # Key/Value: Ž (backward states) -> d_model
        self.key_proj = nn.Linear(keep_dim, d_model)
        self.value_proj = nn.Linear(keep_dim, d_model)

        # ============================================================
        # 2. Cross Attention Mechanism
        # ============================================================
        self.cross_attn = RoPEMultiHeadAttention(
            n_head=n_head, 
            d_model=d_model, 
            d_k=d_k, 
            d_v=d_v, 
            attn_dropout=dropout,
            max_len=time_dim + 2  # 足够长的 max_len
        )
        
        # ============================================================
        # 3. Output Processing
        # ============================================================
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Position-wise FFN
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        
        # ============================================================
        # 4. Output Projection (Delta + Residual from forward)
        # ============================================================
        # 预测 delta，最终输出 Ž_s = z_s^f + delta
        self.delta_head = nn.Linear(d_model, keep_dim)
        
        # 🔥 Small-Init: 初始化为接近 0，使得初始输出 Ž_s ≈ z_s^f (平滑先验)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)

        # 🔥 Cache for efficient backward pass
        self._cache = {}

    def precompute_forward_features(self, H_full, time_stamps=None):
        """
        在 Backward Pass 开始前，预计算并缓存 forward features。
        
        Args:
            H_full: [B, T, keep_dim] - Forward Pass 的输出 z^f (filter output)
            time_stamps: 时间戳（保留接口兼容性）
        
        调用时机：在 backward loop 开始前调用一次
        """
        B, T, d = H_full.shape
        device = H_full.device
        
        # 预计算 Query embeddings: [B, T, d_model]
        query_emb_full = self.query_proj(H_full)
        
        self._cache = {
            'H_full': H_full,                  # [B, T, keep_dim]
            'query_emb_full': query_emb_full,  # [B, T, d_model]
            'B': B,
            'T': T
        }
        
        # 初始化 backward states 缓存
        # backward_states[i] 存储 Ž_{i+1}（代码下标）= Ž_{i+2}（数学下标）
        # 最终 backward_states 包含 [Ž_1, Ž_2, ..., Ž_T]（代码下标 0 到 T-1）
        self._cache['backward_states'] = [None] * T
        
    def initialize_boundary(self, z_T=None):
        """
        设置边界条件: Ž_T = z_T^f（最后一个时刻用 forward output 初始化）
        
        Args:
            z_T: [B, keep_dim] - 可选的边界值，默认使用 H_full[:, -1, :]
        """
        assert self._cache, "Must call precompute_forward_features first!"
        
        H_full = self._cache['H_full']
        T = self._cache['T']
        
        if z_T is None:
            z_T = H_full[:, -1, :]  # [B, keep_dim]
        
        # 存储边界条件 Ž_{T-1}（代码下标）
        self._cache['backward_states'][T-1] = z_T
        
        return z_T

    def forward_step(self, s, z_f_s=None, noise=None):
        """
        Backward Loop 中的单步计算（使用缓存）
        
        理论公式：Ž_s = BackwardFunc(Ž_{s+1:T}, z_s^f)
        
        Args:
            s: int, 当前时间步索引（代码下标，0-indexed）
               - s = T-2: 生成 Ž_{T-2}，attend to [Ž_{T-1}]（1 个元素）
               - s = T-3: 生成 Ž_{T-3}，attend to [Ž_{T-2}, Ž_{T-1}]（2 个元素）
               - ...
               - s = 0:   生成 Ž_0，attend to [Ž_1, ..., Ž_{T-1}]（T-1 个元素）
            z_f_s: [B, keep_dim] - 可选，当前时刻的 forward feature
                   如果为 None，从缓存中取 H_full[:, s, :]
            noise: 可选的噪声输入（用于 stochastic generation）
        
        Returns:
            z_s: [B, keep_dim] - Ž_s（当前时刻的 smoothed state）
        """
        assert self._cache, "Must call precompute_forward_features first!"
        
        B = self._cache['B']
        T = self._cache['T']
        H_full = self._cache['H_full']
        backward_states = self._cache['backward_states']
        device = H_full.device
        
        # ============================================================
        # Step 1: 准备 Query [B, 1, d_model]
        # ============================================================
        # Query = z_s^f（当前时刻的 forward feature）
        if z_f_s is None:
            z_f_s = H_full[:, s, :]  # [B, keep_dim]
        
        query = self._cache['query_emb_full'][:, s:s+1, :]  # [B, 1, d_model]
        
        # ============================================================
        # Step 2: 准备 Key/Value from backward states Ž_{s+1:T}
        # ============================================================
        # 收集所有已生成的 backward states（时间上在 s 之后）
        # backward_states[s+1], backward_states[s+2], ..., backward_states[T-1]
        future_states = []
        for idx in range(s + 1, T):
            if backward_states[idx] is not None:
                future_states.append(backward_states[idx])
        
        if len(future_states) == 0:
            # 边界情况：s = T-1，没有 future states，直接返回 z_f_s
            return z_f_s
        
        # Stack: [B, num_future, keep_dim]
        future_states_tensor = torch.stack(future_states, dim=1)
        num_future = future_states_tensor.shape[1]
        
        # Project to Key/Value: [B, num_future, d_model]
        k = self.key_proj(future_states_tensor)
        v = self.value_proj(future_states_tensor)
        
        # ============================================================
        # Step 3: Cross Attention
        # ============================================================
        # Query: [B, 1, d_model] - z_s^f
        # Key/Value: [B, num_future, d_model] - Ž_{s+1:T}
        # RoPE positions:
        #   - Query position = s
        #   - Key positions = s+1, s+2, ..., T-1
        context, _ = self.cross_attn(
            q=query,
            k=k,
            v=v, 
            attn_mask=None,
            alibi_bias=None,
            start_pos_q=s,          # query 位于时间位置 s
            start_pos_k=s + 1       # key 从时间位置 s+1 开始
        )
        
        # ============================================================
        # Step 4: Residual Connection & Layer Norm & FFN
        # ============================================================
        # Pre-Norm style
        context = context + query  # Residual
        context = self.layer_norm1(context)
        
        # Position-wise FFN with residual
        context_ffn = self.pos_ffn(context)
        context = context + context_ffn
        context = self.layer_norm2(context)
        
        fused = context.squeeze(1)  # [B, d_model]
        
        # ============================================================
        # Step 5: Output (Delta from forward feature)
        # ============================================================
        z_delta = self.delta_head(fused)  # [B, keep_dim]
        
        # 🔥 Final: Ž_s = z_s^f + delta (residual from forward)
        z_s = z_f_s + z_delta
        
        # 数值稳定性
        z_s = torch.clamp(z_s, min=-100.0, max=100.0)
        
        # 🔥 存储到缓存中，供后续步骤使用
        self._cache['backward_states'][s] = z_s
        
        return z_s
    
    def backward_pass(self, H_full=None, return_all=True):
        """
        完整的 Backward Pass: 从 T-1 到 0 逐步生成所有 Ž
        
        Args:
            H_full: [B, T, keep_dim] - Forward Pass 的输出
                    如果为 None，使用缓存中的 H_full
            return_all: bool - 是否返回所有时间步的结果
        
        Returns:
            如果 return_all=True:
                Z_smooth: [B, T, keep_dim] - 所有时间步的 smoothed states
            否则:
                Ž_0: [B, keep_dim] - 最后一个生成的状态
        """
        if H_full is not None:
            self.precompute_forward_features(H_full)
        
        assert self._cache, "Must call precompute_forward_features first!"
        
        T = self._cache['T']
        
        # Step 1: 初始化边界条件 Ž_{T-1} = z_{T-1}^f
        self.initialize_boundary()
        
        # Step 2: Backward loop from T-2 to 0
        for s in range(T - 2, -1, -1):
            self.forward_step(s)
        
        # Step 3: 收集结果
        if return_all:
            backward_states = self._cache['backward_states']
            Z_smooth = torch.stack(backward_states, dim=1)  # [B, T, keep_dim]
            return Z_smooth
        else:
            return self._cache['backward_states'][0]
    
    def forward(self, H, z_next_states, noise=None):
        """
        兼容接口（类似 V4 的 forward）
        
        对于 V6，这个接口用于给定 future states 直接计算当前状态。
        
        Args:
            H: [B, 1, keep_dim] - 当前时刻的 forward feature z_s^f
            z_next_states: [B, num_future, keep_dim] - 所有 future backward states Ž_{s+1:T}
            noise: 可选噪声
            
        Returns:
            z_curr: [B, keep_dim] - Ž_s（当前时刻的 smoothed state）
        """
        B = H.shape[0]
        device = H.device
        
        z_f_s = H.squeeze(1) if H.dim() == 3 else H  # [B, keep_dim]
        
        # 如果没有 future states，直接返回 forward feature
        if z_next_states is None or z_next_states.shape[1] == 0:
            return z_f_s
        
        num_future = z_next_states.shape[1]
        
        # ============================================================
        # Step 1: Query from forward feature
        # ============================================================
        query = self.query_proj(z_f_s).unsqueeze(1)  # [B, 1, d_model]
        
        # ============================================================
        # Step 2: Key/Value from future backward states
        # ============================================================
        k = self.key_proj(z_next_states)   # [B, num_future, d_model]
        v = self.value_proj(z_next_states)  # [B, num_future, d_model]
        
        # ============================================================
        # Step 3: Cross Attention
        # ============================================================
        context, _ = self.cross_attn(
            q=query,
            k=k,
            v=v, 
            attn_mask=None,
            alibi_bias=None,
            start_pos_q=0,  # 相对位置
            start_pos_k=1   # key 从 query 之后开始
        )
        
        # ============================================================
        # Step 4: Residual + LayerNorm + FFN
        # ============================================================
        context = context + query
        context = self.layer_norm1(context)
        context_ffn = self.pos_ffn(context)
        context = context + context_ffn
        context = self.layer_norm2(context)
        
        fused = context.squeeze(1)  # [B, d_model]
        
        # ============================================================
        # Step 5: Output (Delta from forward)
        # ============================================================
        z_delta = self.delta_head(fused)
        z_curr = z_f_s + z_delta
        
        z_curr = torch.clamp(z_curr, min=-100.0, max=100.0)
        
        return z_curr

    def clear_cache(self):
        """清理缓存"""
        self._cache = {}


########### ------------ decoder v2

########### ------------ decoder v2


class SSM_Decoder_V2(nn.Module):
    """
    SSM Decoder (Lite + Attention + FFN)
    
    理论公式: 
        X_t ~ Dec(Z_t, X^o_t, Noise)
        X_gen = Base_Guess + (Pattern(Ctx, Noise) * Scale(Ctx) + Bias(Ctx))
    """
    def __init__(self, feature_dim, z_dim, d_model=64, n_heads=4, noise_dim=32, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.noise_dim = noise_dim
        
        # ======================================================
        # 1. Rich Embeddings (构建上下文原料)
        # ======================================================
        # Base Guess: 仅从 Z 得到的粗略预测，用作 Residual 的基准
        self.z_to_base = nn.Linear(z_dim, feature_dim) 
        
        # Embeddings
        self.res_proj = nn.Linear(1, d_model)              # 残差投影
        self.z_ctx_proj = nn.Linear(z_dim, d_model)        # Z 上下文投影
        self.mask_embedding = nn.Embedding(2, d_model)     # 状态嵌入
        self.position_embedding = nn.Parameter(torch.randn(1, feature_dim, d_model)) # 特征ID嵌入
        
        # ======================================================
        # 2. Transformer Block (特征交互与加工)
        # ======================================================
        # Part A: Attention (Mixin)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(d_model) # Pre-Norm for Attn
        
        # Part B: FFN (Processing)
        # d_ffn 通常设为 d_model 的 2 倍或 4 倍
        self.ffn = PositionWiseFeedForward(d_model, d_model * 2, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)  # Pre-Norm for FFN

        # ======================================================
        # 3. Patient Adapter (估计分布参数)
        # ======================================================
        # 根据加工后的 Context 预测：Scale (波动幅度) 和 Bias (偏移量)
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2) # 输出 [Scale, Bias]
        )
        
        # ======================================================
        # 4. Pattern Generator (隐式分布生成)
        # ======================================================
        self.pattern_proj = nn.Linear(d_model + noise_dim, d_model)
        
        self.backbone = nn.Sequential(
            ResMLPBlock(d_model, dropout),
            ResMLPBlock(d_model, dropout)
        )
        self.head_pattern = nn.Linear(d_model, 1) # 输出归一化的形状

    def forward(self, z_t, x_obs, mask_obs, noise=None):
        """
        z_t: [B, T, z_dim]
        x_obs: [B, T, p]
        mask_obs: [B, T, p]
        """
        B, T, p = x_obs.shape
        
        # Flatten: [N, p, ...] where N = B*T
        # 我们把每个时间步看作独立的样本，把特征维度 p 看作 Sequence Length
        z_flat = z_t.reshape(B*T, -1)     # [N, z_dim]
        x_flat = x_obs.reshape(B*T, p)    # [N, p]
        m_flat = mask_obs.reshape(B*T, p) # [N, p]
        
        # ======================================================
        # Step 1: Rich Context Construction
        # ======================================================
        # 1.1 Base Guess (Trend from Z)
        x_base_guess = self.z_to_base(z_flat) # [N, p]
        
        # 1.2 Residual Calculation
        # Mask=0 (缺失) -> Residual=0
        # Mask=1 (观测) -> Residual=真实值-预测值
        residual = (x_flat - x_base_guess) * m_flat.float()
        
        # 1.3 Embed & Fuse
        # Z 需要unsqueeze并广播到所有特征: [N, 1, d] -> [N, p, d]
        emb_z = self.z_ctx_proj(z_flat).unsqueeze(1).expand(-1, p, -1)
        emb_res = self.res_proj(residual.unsqueeze(-1))           # [N, p, d]
        emb_mask = self.mask_embedding(m_flat.long())             # [N, p, d]
        emb_pos = self.position_embedding.expand(B*T, -1, -1)     # [N, p, d]
        
        context = emb_res + emb_mask + emb_pos + emb_z # [N, p, d]
        
        # ======================================================
        # Step 2: Transformer Encoder Layer (Attn + FFN)
        # ======================================================
        # 2.1 Attention Block (Pre-Norm)
        # Norm -> Attn -> Add
        context_norm = self.attn_norm(context)
        context_h, _ = self.feature_attention(context_norm, context_norm, context_norm)
        context = context + context_h 
        
        # 2.2 FFN Block (Pre-Norm)
        # Norm -> FFN -> Add
        # 🔥 FFN 使得 Context 具有了非线性语义，能更好指导 Adapter
        context_norm = self.ffn_norm(context)
        context_ffn = self.ffn(context_norm)
        context = context + context_ffn
        
        # 此时 context [N, p, d] 包含了丰富的交互信息和语义特征
        
        # ======================================================
        # Step 3: Estimate Scale & Bias (Adapter)
        # ======================================================
        adapter_out = self.adapter(context) # [N, p, 2]
        p_scale, p_bias = adapter_out.chunk(2, dim=-1)
        
        # Scale 必须 > 0，且最好初始化在 1.0 附近 (Softplus(0) + 0.5 ≈ 1.2)
        p_scale = F.softplus(p_scale) + 0.5
        p_scale = p_scale.squeeze(-1)
        p_bias = p_bias.squeeze(-1)
        
        # ======================================================
        # Step 4: Generate Pattern (Implicit)
        # ======================================================
        if noise is None:
            noise = torch.randn(B*T, p, self.noise_dim, device=z_t.device)
            
        # 拼接 Context 和 Noise
        gen_input = torch.cat([context, noise], dim=-1) # [N, p, d+noise]
        
        # ResMLP 生成 normalized pattern
        h = self.backbone(self.pattern_proj(gen_input))
        raw_pattern = self.head_pattern(h).squeeze(-1) # [N, p]
        
        # ======================================================
        # Step 5: Final Composition
        # ======================================================
        # Formula: X = Base + (Pattern * Scale + Bias)
        
        x_delta = raw_pattern * p_scale + p_bias
        x_gen = x_base_guess + x_delta
        
        # Safety Clamp
        x_gen = torch.clamp(x_gen, min=-100.0, max=100.0)
        
        return x_gen.reshape(B, T, p)