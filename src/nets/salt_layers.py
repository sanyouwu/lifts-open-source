"""
Layer modules for self-attention models (Transformer and SAITS).

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ... (Imports) ...
# ... (ScaledDotProductAttention Class) ...
# ... (MultiHeadAttention Class) ...
# ... (PositionWiseFeedForward Class) ...
##################### Rope 
import torch
import torch.nn as nn
class RoPE1D(nn.Module):
    def __init__(self, dim, max_len=48, base=10000.0): # 针对你的短序列调整了默认值
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len
        
        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        """
        🔥 修改点：只处理一个输入 x，并允许指定起始位置
        Args:
            x: [Batch, n_heads, SeqLen, HeadDim] 
            start_pos: 这个 x 在整个序列中的起始位置 (默认 0)
        """
        batch, n_heads, seq_len, head_dim = x.shape
        end_pos = start_pos + seq_len
        
        # 自动扩展缓存 (如果超出了预设长度)
        if end_pos > self.max_len:
            self._set_cos_sin_cache(end_pos + 50) # 多扩一点
            
        # 🔥 关键：根据 start_pos 切片
        # 训练时: start_pos=0, seq_len=T -> 取 [0:T]
        # 推理时: start_pos=t, seq_len=1 -> 取 [t:t+1]
        cos = self.cos_cached[start_pos:end_pos].view(1, 1, seq_len, head_dim)
        sin = self.sin_cached[start_pos:end_pos].view(1, 1, seq_len, head_dim)
        
        return self._apply_rotary_pos_emb(x, cos, sin)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotate_half(x) * sin)

# ============================================================
# 🔥 ALiBi (Attention with Linear Biases) - 用于时序平滑
# ============================================================

def get_alibi_slopes(num_heads):
    """
    获取 ALiBi 每个 attention head 的斜率。
    遵循 ALiBi 论文: slopes = 2^(-8/num_heads * i) for i = 1..num_heads
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = base ** powers
    
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
        extra_slopes = extra_base ** extra_powers
        slopes = torch.cat([slopes, extra_slopes])
    
    return slopes


def get_alibi_bias(seq_len, num_heads, device):
    """
    生成 ALiBi bias 矩阵 (用于 self-attention).
    
    Args:
        seq_len: int, 序列长度
        num_heads: int, 注意力头数
        device: torch device
    
    Returns:
        alibi_bias: [num_heads, seq_len, seq_len] - 添加到 attention scores
    """
    slopes = get_alibi_slopes(num_heads).to(device)
    
    # 位置差矩阵: [seq_len, seq_len]
    positions = torch.arange(seq_len, device=device)
    pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
    
    # ALiBi bias: slopes * (i - j), 使用有符号距离
    # Shape: [num_heads, seq_len, seq_len]
    alibi_bias = slopes.unsqueeze(1).unsqueeze(2) * pos_diff.unsqueeze(0)
    
    return alibi_bias


def get_alibi_bias_1d(seq_len, num_heads, device, query_pos=None):
    """
    生成 1D ALiBi bias (用于 cross-attention，单个 query 位置).
    
    Args:
        seq_len: int, key 序列长度
        num_heads: int, 注意力头数
        device: torch device
        query_pos: int or None, query 位置，如果 None 则默认为 0
    
    Returns:
        alibi_bias: [num_heads, 1, seq_len] - 单个 query 对 seq_len keys 的 bias
    """
    slopes = get_alibi_slopes(num_heads).to(device)
    
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    if query_pos is not None:
        # 从 query_pos 到每个 key 位置的距离
        pos_diff = positions - query_pos
    else:
        # 从位置 0 开始的距离
        pos_diff = positions
    
    # ALiBi bias: [num_heads, seq_len]
    alibi_bias = slopes.unsqueeze(1) * pos_diff.unsqueeze(0)
    
    # 添加 query 维度: [num_heads, 1, seq_len]
    return alibi_bias.unsqueeze(1)


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention with optional ALiBi support"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None, alibi_bias=None):
        """
        Args:
            q, k, v: [B, n_head, len_q/len_k, d_k/d_v]
            attn_mask: optional attention mask
            alibi_bias: optional ALiBi bias [1, n_head, len_q, len_k] or broadcastable
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        # 🔥 在 mask 和 softmax 之前应用 ALiBi bias
        if alibi_bias is not None:
            attn = attn + alibi_bias
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention with optional ALiBi support"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None, alibi_bias=None):
        """
        Args:
            q: [B, len_q, d_model]
            k: [B, len_k, d_model]
            v: [B, len_v, d_model]
            attn_mask: optional mask
            alibi_bias: optional ALiBi bias [n_head, len_q, len_k] 或 [1, n_head, len_q, len_k]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        # 🔥 处理 ALiBi bias 形状: [n_head, len_q, len_k] -> [1, n_head, len_q, len_k]
        if alibi_bias is not None and alibi_bias.dim() == 3:
            alibi_bias = alibi_bias.unsqueeze(0)

        v, attn_weights = self.attention(q, k, v, attn_mask, alibi_bias=alibi_bias)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights

###################------------- add rope--multihead attention
class RoPEMultiHeadAttention(nn.Module):
    """
    Original Transformer multi-head attention 
    + Optional ALiBi support 
    + 🔥 RoPE support (New)
    """

    # 1. Init 增加 max_len 用于 RoPE
    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout, max_len=48,rope_base = 2000):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # 🔥 新增: RoPE 模块
        # 注意：RoPE 必须匹配 head_dim (d_k)
        self.rope = RoPE1D(dim=d_k, max_len=max_len, base=rope_base)

        # 复用你原来的 Attention 计算逻辑
        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    # 2. Forward 增加 start_pos 参数
    def forward(self, q, k, v, attn_mask=None, alibi_bias=None, start_pos_q=0, start_pos_k=0):
        """
        Args:
            q: [B, len_q, d_model]
            k: [B, len_k, d_model]
            v: [B, len_v, d_model]
            attn_mask: optional mask
            alibi_bias: optional ALiBi bias
            start_pos_q: 🔥 Q 在时间轴的起始位置
            start_pos_k: 🔥 K 在时间轴的起始位置
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Projections
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose: [B, n_head, Len, Dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # ==========================================================
        # 🔥🔥🔥 只有这里改了：插入 RoPE 🔥🔥🔥
        # ==========================================================
        # 在进入 self.attention 之前，旋转 Q 和 K
        q = self.rope(q, start_pos=start_pos_q)
        k = self.rope(k, start_pos=start_pos_k)
        # ==========================================================

        # --- 以下完全保留你原版的 Mask 处理逻辑 ---
        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, 
            # so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)

        # --- 以下完全保留你原版的 ALiBi 处理逻辑 ---
        # 🔥 处理 ALiBi bias 形状: [n_head, len_q, len_k] -> [1, n_head, len_q, len_k]
        if alibi_bias is not None and alibi_bias.dim() == 3:
            alibi_bias = alibi_bias.unsqueeze(0)

        # 计算 Attention
        v, attn_weights = self.attention(q, k, v, attn_mask, alibi_bias=alibi_bias)

        # Output Projection
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


####### original causal encoder layer
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,  
        **kwargs
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs["diagonal_attention_mask"]
        self.device = kwargs["device"]
        self.d_time = d_time
        self.d_feature = d_feature
        # print("EncoderLayer d_model:", d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        ##### add a projection layer if d_output is specified
        # self.d_output = d_output
        # if self.d_output is not None and self.d_output != d_model:
        #     self.output_projection = nn.Linear(d_model, d_output)
        # else:
        #     # If d_output is not specified or equals d_model, no projection is needed
        #     self.output_projection = nn.Identity()
    def forward(self, enc_input):
        # print("111111111enc_input shape:", enc_input.shape)
        current_T = enc_input.size(1) # 获取当前的序列长度 (T-1)
        if self.diagonal_attention_mask:
            # mask_time = torch.eye(self.d_time).to(self.device)
            #### changed to upper triangular matrix without diagonal
            #### 允许看到自己，因为seq_len = T - 1，第一个我单独补上; 
            # 🔥 关键修复：使用 enc_input.device 而不是 self.device，确保 mask 和输入在同一设备
            mask_time = ~torch.tril(torch.ones(current_T, current_T)).bool().to(enc_input.device)
            ###### extactly forecasting, use this with masked_fill() function
            # mask_time = ~torch.tril(torch.ones(self.d_time, self.d_time), diagonal = -1).bool().to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        # print("2222222enc_input shape:", enc_input.shape)
        enc_input = self.layer_norm(enc_input)
        # print("333333after layer norm:", enc_input.shape)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        #### add a output projection layer
        # enc_output = self.output_projection(enc_output) # Shape: (B, T-1, d_output) or (B, T, d_output) depending on input
        return enc_output, attn_weights

##########-- rope version of encoder layer
##########-- rope version of encoder layer
##########-- rope version of encoder layer
##########-- rope version of encoder layer
##########-- rope version of encoder layer
class RoPEEncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        max_len=50,  # 🔥 新增: RoPE 需要预计算的最大长度
        **kwargs
    ):
        super(RoPEEncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs.get("diagonal_attention_mask", True) # 默认为 True 以防万一
        self.d_time = d_time
        self.d_feature = d_feature
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 🔥 修改: 替换为 RoPEMultiHeadAttention
        self.slf_attn = RoPEMultiHeadAttention(
            n_head=n_head, 
            d_model=d_model, 
            d_k=d_k, 
            d_v=d_v, 
            attn_dropout=attn_dropout,
            max_len=max_len # 传入 max_len
        )
        
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, start_pos=0): # 🔥 新增: start_pos 参数
        """
        enc_input: [Batch, T, D]
        start_pos: 当前序列在时间轴上的起始位置 (通常训练/Filtering时为0)
        """
        current_T = enc_input.size(1) 
        
        # --- 1. Mask 生成逻辑 (保留原汁原味) ---
        if self.diagonal_attention_mask:
            # 生成上三角掩码 (True 代表被 Mask 掉的位置)
            # [T, T]
            mask_time = ~torch.tril(torch.ones(current_T, current_T)).bool().to(enc_input.device)
        else:
            mask_time = None

        residual = enc_input
        
        # --- 2. Pre-LN ---
        enc_input = self.layer_norm(enc_input)
        
        # --- 3. Attention (with RoPE) ---
        # Self-Attention: Q, K, V 都是 enc_input
        # 这里的 start_pos_q 和 start_pos_k 都是 start_pos
        # 因为在 Self-Attention 中，Q 和 K 是同一个序列，完全对齐
        enc_output, attn_weights = self.slf_attn(
            q=enc_input, 
            k=enc_input, 
            v=enc_input, 
            attn_mask=mask_time,
            alibi_bias=None,   # 不用 ALiBi
            start_pos_q=start_pos, # 🔥 告诉 RoPE 绝对位置
            start_pos_k=start_pos
        )
        
        enc_output = self.dropout(enc_output)
        enc_output += residual

        # --- 4. FFN ---
        enc_output = self.pos_ffn(enc_output)
        
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        ### d_hid = d_model in the original Transformer paper
        ### n_position = max_len in the original Transformer paper
        
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()
