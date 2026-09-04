import torch
################## B x p.  x T

def compute_masked_l2_distance(x, y, obs_mask=None, obs_mask_prime=None,time_weights=None, div = None, lambda_fo=1.0, lambda_fx=1.0, fo_loss_type='l2'):
    """
    计算 L2 距离。
    
    模式 1 (传统 Masked L2): 
        如果只传入 obs_mask，则计算 || (x - y) * obs_mask ||。
        
    模式 2 (扩展距离 metric d_alpha):
        如果传入了 obs_mask AND obs_mask_prime，则不再用 mask 过滤数据，
        而是计算 lambda_fx * sqrt( ||x - y||^2 ) + lambda_fo * sqrt( ||mask - mask'||^2 )。
        这通常用于 Energy Score 计算中比较两个样本（及其对应的 mask）。
    
    Args:
        lambda_fo: float, weight for mask (fo) loss, default 1.0
                   Set to 0 to disable mask loss, >1 to emphasize mask prediction
        lambda_fx: float, weight for data (fx) loss, default 1.0
                   Set to 0 to train only on mask prediction (isolate mask impact)
        fo_loss_type: str, 'l2' (default) or 'bce'
                   'l2':  mask loss = sqrt(sum_P((o - o')^2))  (original)
                   'bce': mask loss = mean_P(BCE(o, o'))  (binary cross-entropy)
    """
    
    # 1. [基础] 将 NaN 填充为 0
    x_filled = torch.nan_to_num(x, nan=0.0)
    y_filled = torch.nan_to_num(y, nan=0.0)
    
    # =========================================================================
    # 分支 A: 双 Mask 模式 (d_alpha 公式)
    # 当比较两个样本及其 Mask 时，Mask 本身也是被比较的对象，而不是过滤器
    # =========================================================================
    if obs_mask is not None and obs_mask_prime is not None:
        
        # --- 1. 对齐 Device ---
        if obs_mask.device != x.device: obs_mask = obs_mask.to(x.device)
        if obs_mask_prime.device != x.device: obs_mask_prime = obs_mask_prime.to(x.device)
            
        # --- 2. 转换类型 ---
        obs_mask = obs_mask.float()
        obs_mask_prime = obs_mask_prime.float()
        
        # --- 3. 维度广播处理 (Broadcasting) ---
        # x 可能有 Group 维度 (B, G, P, T)，而 mask 可能是 (B, P, T)
        if x.ndim == 4 and obs_mask.ndim == 3:
            obs_mask = obs_mask.unsqueeze(1)
        if x.ndim == 4 and obs_mask_prime.ndim == 3:
            obs_mask_prime = obs_mask_prime.unsqueeze(1)
            
        # 【核心操作】
        # Truth: 如果 m=0, 这里的 0.0 * 0 = 0. (解决了 NaN-Value 问题)
        # Gen:   如果 m'=0.8, 这里就是 x' * 0.8 (保留了梯度的 Soft Gating)
        ##########---- try not to mask data ----##########

        # --- 4. 计算两部分差异 ---
        # Part 1: 数据差异 sum((x - x')^2)
        # 注意：这里不再乘 mask，直接算 x 和 y 的距离 (假设 x, y 在 mask=0 处已经是 0 或者填补值)
        # diff_data_sq = (x_filled - y_filled) ** 2  
        # sum_data_sq = torch.sum(diff_data_sq, dim=-2) # Sum over P
        
        # Part 2: Mask 差异 sum((o - o')^2)
        diff_mask_sq = (obs_mask - obs_mask_prime) ** 2
        sum_mask_sq = torch.sum(diff_mask_sq, dim=-2) # Sum over P
        
        ### add part 3 ---updated kaizheng loss --- keep interaction between data and mask
        # x_filled = x_filled * obs_mask
        # y_filled = y_filled * obs_mask_prime
        if div is not None:
            diff_data_sq_inter = (x_filled * div  - y_filled * div) ** 2
            # diff_data_sq_inter = (x_filled   - y_filled) ** 2
        else:
            diff_data_sq_inter = (x_filled * obs_mask - y_filled * obs_mask) ** 2
            # diff_data_sq_inter = (x_filled  - y_filled) ** 2
        sum_mask_sq_inter = torch.sum(diff_data_sq_inter, dim=-2) # Sum over P
        # --- 
       
        # result = sqrt( sum_data + sum_mask )
        ##### 5. 合并开根号 ---
        # l2_norm_p = torch.sqrt(sum_mask_sq_inter + 1 * sum_mask_sq + 1e-8)
         #### or 开根号 再合并？
        # 🔥 lambda_fo 控制 mask loss 的权重
        if fo_loss_type == 'bce':
            # Binary cross-entropy for mask prediction
            # Clamp to avoid log(0); obs_mask is binary {0,1}, obs_mask_prime is sigmoid output in (0,1)
            o_p = obs_mask_prime.clamp(1e-6, 1 - 1e-6)
            o_t = obs_mask.clamp(1e-6, 1 - 1e-6)
            bce = -(o_t * torch.log(o_p) + (1 - o_t) * torch.log(1 - o_p))
            mask_loss_p = torch.mean(bce, dim=-2)  # mean over P → (B, [G], T)
            l2_norm_p = lambda_fx * torch.sqrt(sum_mask_sq_inter + 1e-8) + lambda_fo * mask_loss_p
        else:
            # Original L2: sqrt(sum_P((o - o')^2))
            l2_norm_p = lambda_fx * torch.sqrt(sum_mask_sq_inter + 1e-8) + lambda_fo * torch.sqrt(sum_mask_sq + 1e-8)
        #### triple is wrong
        # l2_norm_p = torch.sqrt(sum_data_sq + 1 * sum_mask_sq + 1e-8 + 1 * sum_mask_sq_inter)
    # =========================================================================
    # 分支 B: 单 Mask 模式 (传统 Masked L2)
    # Mask 用作过滤器，只计算观测点的距离
    # =========================================================================
    else:
        # 1. 生成基础 NaN Mask
        nan_mask = (~torch.isnan(x)) & (~torch.isnan(y))
        nan_mask = nan_mask.float()
        
        # 2. 融合 obs_mask (如果存在)
        if obs_mask is not None:
            if obs_mask.device != x.device: obs_mask = obs_mask.to(x.device)
            obs_mask = obs_mask.float()
            
            if x.ndim == 4 and obs_mask.ndim == 3:
                obs_mask = obs_mask.unsqueeze(1)
                
            final_mask = nan_mask * obs_mask
        else:
            final_mask = nan_mask
            
        # 3. 应用 Mask 并计算差值
        diff = (x_filled - y_filled) * final_mask
        
        # 4. 计算 P 维度 L2 Norm
        l2_norm_p = torch.sqrt(torch.sum(diff ** 2, dim=-2) + 1e-8)

    # =========================================================================
    # 后处理: 时间和 Batch 维度平均
    # =========================================================================
    
    # 5. 应用时间权重 (保留你的逻辑结构，虽然这里被注释掉了)
    # if time_weights is not None:
    #     if time_weights.device != x.device:
    #         time_weights = time_weights.to(x.device)
    #     dist_mean_t = l2_norm_p * time_weights.unsqueeze(0)
    # else:
    
    # 在 T 维度 (dim=-1) 取平均
    dist_mean_t = torch.mean(l2_norm_p, dim=-1)
    
    # 6. 对剩下的维度 (Batch, Group) 取平均
    loss = torch.mean(dist_mean_t)
    
    return loss
#### version 1
# def compute_masked_l2_distance(x, y, time_weights = None, obs_mask=None):
#     """
#     计算 Masked L2 距离。
#     输入可以是 (B, P, T) 或 (B, G, P, T)。
#     逻辑：NaN填0 -> 差值 -> 掩码 -> P维L2范数 -> T维平均 -> 剩余维度平均
#     """
    
#     # 1. [关键] 将 NaN 填充为 0
#     # 这一步保证了 x - y 计算时不会出现 NaN
#     x_filled = torch.nan_to_num(x, nan=0.0)
#     y_filled = torch.nan_to_num(y, nan=0.0)
    
#     # 2. 生成掩码 Mask
#     # 只有当 x 和 y 都有值时，才计算距离
#     nan_mask = (~torch.isnan(x)) & (~torch.isnan(y))
#     nan_mask = nan_mask.float()
    
# # 3. [核心修改] 融合传入的 obs_mask
#     if obs_mask is not None:
#         # 确保 device 一致
#         if obs_mask.device != x.device:
#             obs_mask = obs_mask.to(x.device)
            
#         # 确保是 float 类型以便相乘
#         obs_mask = obs_mask.float()
        
#         # 维度对齐处理 (Broadcasting)
#         # 如果 x 是 (B, G, P, T) 而 mask 是 (B, P, T)，需要在 G 维度unsqueeze
#         if x.ndim == 4 and obs_mask.ndim == 3:
#             # 假设 mask 是 (B, P, T), x 是 (B, G, P, T)
#             # 变为 (B, 1, P, T) 以便广播
#             obs_mask = obs_mask.unsqueeze(1)
            
#         # 【交集逻辑】有效点 = (不是NaN) AND (obs_mask是1)
#         final_mask = nan_mask * obs_mask
#     else:
#         # 如果没有传入 mask，默认只过滤 NaN
#         final_mask = nan_mask
#     # 3. 计算差值并应用 Mask
#     # diff 在缺失位置会变成 (0 - 0) = 0 或者 (0 - value)，
#     # 所以必须乘 mask 强制把缺失位置的差值变成 0，防止模型学习去预测 0
#     diff = (x_filled - y_filled) * final_mask
    
#     # 4. 在 P 维度 (dim=-2) 计算 L2 Norm
#     # 公式: sqrt( sum(diff^2) )
#     # 加 1e-8 是为了防止 sqrt(0) 导致反向传播梯度 NaN (Exploding Gradient)
#     ################ >>>>>>>>> p > 2
#     l2_norm_p = torch.sqrt(torch.sum(diff ** 2, dim=-2) + 1e-8)
#     dist_mean_t = l2_norm_p.mean(dim=-1)
#     ################ >>>>>>>>> p = 2 --- consider whole T (vectorize long vector)
#     # l2_norm_p = torch.sqrt(torch.sum(diff ** 2, dim=[-2,-1]) + 1e-8)
#     # dist_mean_t = l2_norm_p
#     #################
#     # 4. 应用时间权重
#     # if time_weights is not None:
#     #     # time_weights: (T,) -> (1, T) 广播
#     #     # 确保权重在同一个 device 上
#     #     if time_weights.device != x.device:
#     #         time_weights = time_weights.to(x.device)
            
#     #     dist_mean_t = l2_norm_p * time_weights.unsqueeze(0)
#     # else:
#     #     dist_mean_t = torch.mean(l2_norm_p, dim=-1)# 5. 在 T 维度 (dim=-1) 取平均
#     # 6. 对剩下的维度 (Batch, Group) 取平均
#     loss = torch.mean(dist_mean_t)
    
#     return loss


def d_alpha(x1, o1, x2, o2, alpha=1.0, eps=1e-8):
    """
    实现方程式 (4.5) 的 Masked Energy Distance d_alpha。
    
    输入:
        x1, o1, x2, o2: 形状为 (B, G, P, T) 的张量
        alpha: 幂指数，默认为 1.0
        eps: 数值稳定性小量
    
    输出:
        标量，表示所有样本对的平均 d_alpha 距离
    """
    o1 = o1.float()
    o2 = o2.float()
    # 1. 计算 O (Mask) 距离 (L1 距离) -> (B, G, P, T)
    o_dist = torch.abs(o1 - o2)
    
    # 2. 计算 X (观测值) 距离
    x_dist = torch.abs(x1 - x2)  # (B, G, P, T)
    if alpha != 1.0:
        x_dist = torch.pow(x_dist + eps, alpha)  # 加上 epsilon 保证稳定性
    
    # 3. 计算 Term 1: (1 - |o_j - o_j'|) * |x_j^o - x_j^o'|^alpha
    term1 = (1.0 - o_dist) * x_dist  # (B, G, P, T)
    
    # 4. 计算 Term 2: |o_j - o_j'|^alpha
    term2 = o_dist
    if alpha != 1.0:
        term2 = torch.pow(term2 + eps, alpha)
    
    # 5. 在 P 维度上求和 → 得到每个 (b, g, t) 的 d_alpha
    #   d_alpha(b,g,t) = Σ_j [term1(b,g,j,t) + term2(b,g,j,t)]
    d_alpha_per_t = (term1 + term2).sum(dim=-2)  # (B, G, T)
    
    # 6. 在 T 维度上取平均 → 得到每个 (b, g) 的平均 d_alpha
    d_alpha_per_bg = d_alpha_per_t.mean(dim=-1)  # (B, G)
    
    # 7. 在 B 和 G 维度上取平均 → 得到最终标量
    return d_alpha_per_bg.mean()  # 标量
    # return d_alpha_per_bg  # 标量
    


def energy_score(x, y, alpha=1):
    """
    计算两个张量之间的能量分数 (Energy Score)。
    与 d_alpha 类似，但在 P 维度上计算 alpha-范数距离，然后对 T 取平均，最后对 B, G (或 B) 取平均。

    Args:
        x: 张量，形状为 (B, G, P, T) 或 (B, P, T)
        y: 张量，形状与 x 相同
        alpha: 距离的 alpha-范数，默认为 2 (欧几里得距离)

    Returns:
        标量，表示能量分数
    """
    if x.shape != y.shape:
        raise ValueError(f"Input shapes must match, got x: {x.shape}, y: {y.shape}")
    
    # 计算 x - y 的差值
    diff = x - y  # (B, G, P, T) or (B, P, T)
    
    # 计算 P 维度上的 alpha-范数距离
    # 对于每个 (b, g, t) 或 (b, t)，计算 P 维向量的范数
    if alpha == 2:
        # 欧几里得距离
        dist_p = torch.norm(diff, p=2, dim=-2)  # (B, G, T) or (B, T)
    elif alpha == 1:
        # 曼哈顿距离
        dist_p = torch.norm(diff, p=1, dim=-2)  # (B, G, T) or (B, T)
    else:
        # 通用 alpha-范数
        dist_p = torch.norm(diff, p=alpha, dim=-2)  # (B, G, T) or (B, T)
    
    # 在 T 维度上取平均
    avg_dist_per_bg = dist_p.mean(dim=-1)  # (B, G) or (B,)
    
    # 在剩余维度上取平均
    return avg_dist_per_bg.mean()  # 标量




# profile
# Qwen3-Coder

import torch

class TimeSeriesNormalizer:
    def __init__(self,epsilon=1e-6):
        """
        Args:
            epsilon: 防止除以0的微小数值
        """
        self.epsilon = epsilon
        self.global_max = None
        self.global_min = None
        self.device = None
# 
    def fit(self, x_list):
            """
            计算全局的 Max 和 Min (兼容所有 PyTorch 版本)
            """
            # 1. 拼接所有数据
            full_tensor = torch.cat(x_list, dim=1) 
            self.device = full_tensor.device
            
            # 生成 NaN 掩码 (True 表示是 NaN)
            is_nan = torch.isnan(full_tensor)

            # ==============================
            # 计算 Global Max
            # ==============================
            # 复制一份数据用于计算 max
            temp_max = full_tensor.clone()
            # 将 NaN 填充为 -inf (负无穷)，这样它们永远不会被选为 max
            temp_max[is_nan] = -float('inf')
            # torch.max 返回 (values, indices)，我们需要 [0] 取 values
            max_val = torch.max(temp_max, dim=1, keepdim=True)[0]

            # ==============================
            # 计算 Global Min
            # ==============================
            # 复制一份数据用于计算 min
            temp_min = full_tensor.clone()
            # 将 NaN 填充为 +inf (正无穷)，这样它们永远不会被选为 min
            temp_min[is_nan] = float('inf')
            min_val = torch.min(temp_min, dim=1, keepdim=True)[0]

            # ==============================
            # 异常值处理 (防止全 NaN 行导致的 inf)
            # ==============================
            # 如果某一行全是 NaN，max 会是 -inf，min 会是 inf。我们需要修正它。
            # 如果没有 torch.nan_to_num (旧版本)，我们手动处理
            
            # 处理 Max 中的 inf/-inf/nan
            mask_inf = torch.isinf(max_val) | torch.isnan(max_val)
            max_val[mask_inf] = 1.0 # 默认值

            # 处理 Min 中的 inf/-inf/nan
            mask_inf = torch.isinf(min_val) | torch.isnan(min_val)
            min_val[mask_inf] = 0.0 # 默认值

            self.global_max = max_val
            self.global_min = min_val
            
            print(f"Stats computed. Max shape: {self.global_max.shape}")

    def normalize(self, x_input):
        """
        执行归一化到 [-1, 1]，并将原本的 NaN 填充为 0
        支持单个 Tensor 或 List[Tensor]
        """
        if isinstance(x_input, list):
            return [self._norm_single(x) for x in x_input]
        return self._norm_single(x_input)

    def denormalize(self, x_input):
        """
        反归一化回到原始尺度
        """
        if isinstance(x_input, list):
            return [self._denorm_single(x) for x in x_input]
        return self._denorm_single(x_input)

    def _norm_single(self, x):
        # x shape: (P, T)
        # Formula: 2 * (x - min) / (max - min) - 1
        
        numerator = x - self.global_min
        denominator = self.global_max - self.global_min + self.epsilon
        
        x_norm = 2 * (numerator / denominator) - 1
        
        # 关键步骤：将原始数据中的 NaN 填补为 0
        # 在 [-1, 1] 的归一化体系里，0 代表均值/中间状态
        x_norm = torch.nan_to_num(x_norm, nan=0.0)
        
        return x_norm

    def _denorm_single(self, x):
        # Formula: (x + 1) / 2 * (max - min) + min
        
        # 这里的 x 可能已经是没有 NaN 的预测值了
        scale = self.global_max - self.global_min + self.epsilon
        x_orig = ((x + 1) / 2) * scale + self.global_min
        
        return x_orig

# ==========================================
# 使用示例 (Usage Example)
# ==========================================

# 1. 模拟数据：List 中有两个 (3, T) 的矩阵，包含 Na
# X_list[0]
# from modules.criterion import TimeSeriesNormalizer  
# P = 3
# T1, T2 = 5, 4
# data_list = [
#     torch.randn(P, T1),
#     torch.randn(P, T2)
# ]
# # 强行加一些 NaN
# data_list[0][0, 2] = float('nan')
# data_list[1][1, 0] = float('nan')

# print("原始数据 Sample 0 (Row 0):", data_list[0][0])

# # 2. 初始化并计算统计量
# normalizer = TimeSeriesNormalizer()
# normalizer.fit(data_list)

# # 3. 归一化
# norm_list = normalizer.normalize(data_list)
# print("\n归一化后 Sample 0 (Row 0):", norm_list[0][0])
# print("  -> 注意: 原来的 NaN 变成了 0.0")
# print("  -> 注意: 数值都在 [-1, 1] 之间")

# # 4. 模拟模型预测 (假设模型输出了 [-1, 1] 之间的值，且没有 NaN)
# model_output = norm_list # 这里直接假装那是模型输出

# # 5. 反归一化
# restored_list = normalizer.denormalize(model_output)
# print("\n还原后 Sample 0 (Row 0):", restored_list[0][0])

# # 验证误差 (忽略 NaN 位置)
# mask = ~torch.isnan(data_list[0])
# diff = (data_list[0][mask] - restored_list[0][mask]).abs().sum()
# print(f"\n还原误差 (应接近 0): {diff.item():.6f}")