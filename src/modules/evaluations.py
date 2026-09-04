"""
Baseline 方法评估模块 (纯 Numpy 版本，无 torch 依赖)

提供统一的评估函数，适用于 TensorFlow-only 环境下的 baseline 方法 (MIDAS, GP-VAE 等)。
只要生成了 imputation 样本 (N, m, p, T)，即可调用 evaluate_baseline_imputation() 计算所有指标。

与 modules/evaluations_baselines.py 接口完全一致，但不依赖 torch。
"""

import numpy as np
import time
from typing import Dict, List, Optional, Union


def compute_energy_score_simple(samples: np.ndarray, 
                                 target: np.ndarray, 
                                 mask: np.ndarray,
                                 n_pairs: int = 50) -> tuple:
    """
    Energy Score 计算 (per-point average, 与 MAE/CRPS 量级一致)
    
    ES_raw = mean over eval points of [ E[|X - y|] - 0.5 * E[|X - X'|] ]
    ES_norm = ES_raw / mean(|target|)
    
    注意: 使用 L1 距离 (绝对值) 而非 L2 范数，使得 ES 与 MAE 量级一致
    
    Args:
        samples: (N, m, p, T) imputation 样本
        target: (N, p, T) ground truth
        mask: (N, p, T) bool or float, 1=要评估的区域
        n_pairs: 用于估计 E[|X - X'|] 的样本对数量
    
    Returns:
        tuple: (es_raw, es_norm) - 越低越好
    """
    N, m, p_dim, T_dim = samples.shape
    
    if mask.dtype != bool:
        mask = mask > 0.5
    
    # CSDI-style denominator: mean(|target|) on masked region
    target_abs_sum = np.sum(np.abs(target) * mask)
    mask_count = np.sum(mask)
    denominator = target_abs_sum / (mask_count + 1e-8) + 1e-8
    
    # 逐点计算 Energy Score (per evaluation point)
    total_es = 0.0
    total_count = 0
    
    for i in range(N):
        for t in range(T_dim):
            for p_idx in range(p_dim):
                if not mask[i, p_idx, t]:
                    continue
                
                # 取出该点的所有样本: (m,)
                samples_ipt = samples[i, :, p_idx, t]
                target_ipt = target[i, p_idx, t]
                
                # Term 1: E[|X - y|] (使用 L1 距离)
                term1 = np.mean(np.abs(samples_ipt - target_ipt))
                
                # Term 2: 0.5 * E[|X - X'|]
                if m > 1:
                    num_sample_pairs = min(n_pairs, m * (m - 1) // 2)
                    idx1 = np.random.randint(0, m, num_sample_pairs)
                    idx2 = np.random.randint(0, m, num_sample_pairs)
                    same_mask = (idx1 == idx2)
                    idx2[same_mask] = (idx2[same_mask] + 1) % m
                    
                    diff_pairs = np.abs(samples_ipt[idx1] - samples_ipt[idx2])
                    term2 = np.mean(diff_pairs) / 2
                else:
                    term2 = 0.0
                
                total_es += (term1 - term2)
                total_count += 1
    
    es_raw = total_es / (total_count + 1e-8)
    es_norm = es_raw / denominator
    
    return es_raw, es_norm


def compute_crps_quantile(samples: np.ndarray, 
                          target: np.ndarray, 
                          mask: np.ndarray,
                          n_quantiles: int = 100) -> tuple:
    """
    CRPS 计算 (Quantile-based, CSDI 风格)
    
    CRPS = mean over eval points of [ mean over q in quantiles of 2 * rho_q(y, Q_q) ]
    where rho_q is the quantile loss (pinball loss)
    
    Args:
        samples: (N, m, p, T) imputation 样本
        target: (N, p, T) ground truth
        mask: (N, p, T) bool or float, 1=要评估的区域
        n_quantiles: 分位数个数
    
    Returns:
        tuple: (crps_raw, crps_norm, denominator) - 越低越好
    """
    N, m, p_dim, T_dim = samples.shape
    
    if mask.dtype != bool:
        mask = mask > 0.5
    
    # CSDI-style denominator: sum(|target|) / count
    target_abs_sum = np.sum(np.abs(target) * mask)
    mask_count = np.sum(mask)
    denominator = target_abs_sum / (mask_count + 1e-8) + 1e-8
    
    # Quantile levels
    quantiles = np.linspace(0.05, 0.95, n_quantiles)
    
    # 计算分位数 (N, n_q, p, T)
    q_values = np.quantile(samples, quantiles, axis=1)  # (n_q, N, p, T)
    q_values = q_values.transpose(1, 0, 2, 3)  # (N, n_q, p, T)
    
    # 扩展 target: (N, 1, p, T)
    target_expanded = target[:, np.newaxis, :, :]
    
    # Pinball loss for each quantile
    # rho_q(y, q) = (y - q) * (tau - I(y < q))
    #             = tau * (y - q) if y >= q
    #             = (tau - 1) * (y - q) if y < q
    diff = target_expanded - q_values  # (N, n_q, p, T)
    
    total_crps = 0.0
    for i, q in enumerate(quantiles):
        pinball = np.where(diff[:, i, :, :] >= 0, 
                          q * diff[:, i, :, :], 
                          (q - 1) * diff[:, i, :, :])
        # 只在 mask 区域计算
        total_crps += np.sum(pinball * mask)
    
    crps_raw = 2 * total_crps / (mask_count * n_quantiles + 1e-8)
    crps_norm = crps_raw / denominator
    
    return crps_raw, crps_norm, denominator


def compute_coverage(samples: np.ndarray, 
                     target: np.ndarray, 
                     mask: np.ndarray,
                     alpha: float) -> float:
    """
    Coverage 计算
    
    Coverage = fraction of eval points where target is within [q_lo, q_hi]
    
    Args:
        samples: (N, m, p, T) imputation 样本
        target: (N, p, T) ground truth
        mask: (N, p, T) bool or float, 1=要评估的区域
        alpha: confidence level (e.g., 0.9 for 90% CI)
    
    Returns:
        float: coverage rate
    """
    if mask.dtype != bool:
        mask = mask > 0.5
    
    q_lo = (1 - alpha) / 2
    q_hi = 1 - q_lo
    
    lo = np.quantile(samples, q_lo, axis=1)  # (N, p, T)
    hi = np.quantile(samples, q_hi, axis=1)  # (N, p, T)
    
    in_interval = (target >= lo) & (target <= hi)  # (N, p, T)
    
    coverage = np.sum(in_interval * mask) / (np.sum(mask) + 1e-8)
    
    return coverage


def evaluate_baseline_imputation(
    imp_samples: np.ndarray,
    ground_truth: np.ndarray,
    target_mask: np.ndarray,
    method_name: str = "Baseline",
    alphas: Optional[List[float]] = None,
    print_results: bool = True,
    scaler_info: Optional[Dict] = None,
    input_format: str = 'pT',
    verbose: bool = True,
    skip_wasserstein: bool = True,
) -> Dict:
    """
    评估 Baseline 方法的 Imputation 结果 (统一接口, 纯 Numpy 版本)
    
    与 modules/evaluations_baselines.py 的 evaluate_baseline_imputation 接口一致。
    
    Args:
        imp_samples: (N, m, p, T) 或 (N, m, T, p) imputation 样本
        ground_truth: (N, p, T) 或 (N, T, p) ground truth
        target_mask: (N, p, T) 或 (N, T, p) 评估目标 mask, 1=评估点, 0=非评估点
        method_name: 方法名称 (用于输出)
        alphas: coverage 计算的 alpha 列表，默认 [0.2, 0.3, ..., 0.9]
        print_results: 是否打印结果
        scaler_info: {'mean': (p,), 'std': (p,)} 用于 CI Width 的原始尺度计算
        input_format: 'pT' (默认) 或 'Tp'，数据格式
        verbose: 同 print_results (兼容旧接口)
        skip_wasserstein: 是否跳过 Wasserstein 计算 (兼容旧接口)
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    start_time = time.time()
    
    # 处理 verbose 参数 (兼容旧接口)
    if not verbose:
        print_results = False
    
    # 确保是 numpy array
    imp_samples = np.asarray(imp_samples)
    ground_truth = np.asarray(ground_truth)
    target_mask = np.asarray(target_mask)
    
    # 统一为 (N, m, p, T) 格式
    if input_format == 'Tp':
        imp_samples = imp_samples.transpose(0, 1, 3, 2)
        ground_truth = ground_truth.transpose(0, 2, 1)
        target_mask = target_mask.transpose(0, 2, 1)
    
    # 默认 alphas
    if alphas is None:
        alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    N, m, p, T = imp_samples.shape
    
    # 评估区域 = target_mask=1 的位置
    eval_mask = target_mask > 0.5  # (N, p, T) bool
    eval_mask_float = target_mask.astype(np.float32)
    
    # 计算 mean imputation
    imp_mean = imp_samples.mean(axis=1)  # (N, p, T)
    
    # ============================================================
    # 1. MAE 和 RMSE
    # ============================================================
    mae_raw = np.abs(imp_mean - ground_truth)[eval_mask].mean()
    rmse_raw = np.sqrt(((imp_mean - ground_truth)[eval_mask] ** 2).mean())
    
    # ============================================================
    # 2. CRPS (Quantile-based)
    # ============================================================
    crps_raw, crps_norm, denominator = compute_crps_quantile(imp_samples, ground_truth, eval_mask)
    
    # ============================================================
    # 3. Coverage
    # ============================================================
    coverage_list = [compute_coverage(imp_samples, ground_truth, eval_mask, alpha) for alpha in alphas]
    
    # ============================================================
    # 4. CI Width (支持原始尺度)
    # ============================================================
    if scaler_info is not None:
        mean = np.array(scaler_info['mean']).reshape(1, -1, 1)
        std = np.array(scaler_info['std']).reshape(1, -1, 1)
        samples_raw = imp_samples * std + mean  # 反标准化
    else:
        samples_raw = imp_samples
    
    ciw_list = []
    for alpha in alphas:
        q_lo, q_hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
        lo = np.quantile(samples_raw, q_lo, axis=1)  # (N, p, T)
        hi = np.quantile(samples_raw, q_hi, axis=1)  # (N, p, T)
        width = (hi - lo) * eval_mask_float
        ciw = width.sum() / (eval_mask_float.sum() + 1e-8)
        ciw_list.append(ciw)
    
    # ============================================================
    # 5. Energy Score
    # ============================================================
    es_raw, es_norm = compute_energy_score_simple(imp_samples, ground_truth, eval_mask)
    
    # ============================================================
    # 6. MAE/RMSE Norm
    # ============================================================
    mae_norm = mae_raw / denominator
    rmse_norm = rmse_raw / denominator
    
    elapsed = time.time() - start_time
    eval_rate = eval_mask.mean()
    
    # ============================================================
    # 打印结果
    # ============================================================
    if print_results:
        print("=" * 70)
        print(f"{method_name.upper()} IMPUTATION EVALUATION RESULTS (completed in {elapsed:.1f}s)")
        print("=" * 70)
        print(f"Samples: N={N}, Imputations={m}")
        print(f"Data Shape: p={p}, T={T}")
        print(f"Eval Rate: {eval_rate:.1%}")
        
        # Metrics (ES, MAE, CRPS) - Raw and Normalized side by side
        print(f"\n>>> Metrics (ES, MAE, CRPS)")
        print(f"{'Metric':<12} | {'Raw':<14} | {'Normalized':<14}")
        print("-" * 46)
        print(f"{'ES':<12} | {es_raw:>14.4f} | {es_norm:>14.4f}")
        print(f"{'MAE':<12} | {mae_raw:>14.4f} | {mae_norm:>14.4f}")
        print(f"{'CRPS':<12} | {crps_raw:>14.4f} | {crps_norm:>14.4f}")
        
        # Coverage & CI Width (all alphas)
        print(f"\n>>> Coverage & CI Width (all alphas)")
        header = f"{'Alpha':<8} | {'Coverage':<12} | {'Target':<10} | {'CI Width':<12}"
        print(header)
        print("-" * len(header))
        for i, alpha in enumerate(alphas):
            target_cov = alpha * 100
            print(f"{alpha*100:>5.0f}%   | {coverage_list[i]*100:>10.1f}%  | {target_cov:>7.0f}%  | {ciw_list[i]:>12.4f}")
        
        print("=" * 70)
    
    # ============================================================
    # 返回结果字典
    # ============================================================
    results = {
        'method': method_name,
        'MAE_raw': float(mae_raw),
        'RMSE_raw': float(rmse_raw),
        'CRPS_raw': float(crps_raw),
        'MAE_norm': float(mae_norm),
        'RMSE_norm': float(rmse_norm),
        'CRPS_norm': float(crps_norm),
        'ES_raw': float(es_raw),
        'ES_norm': float(es_norm),
        # 兼容旧接口 (不带 _raw 后缀)
        'MAE': float(mae_raw),
        'RMSE': float(rmse_raw),
        'CRPS': float(crps_raw),
        'ES': float(es_raw),
        'coverage': dict(zip(alphas, [float(c) for c in coverage_list])),
        'ci_width': dict(zip(alphas, [float(c) for c in ciw_list])),
        'alphas': alphas,
        'elapsed_time': elapsed,
        'n_samples': N,
        'n_imputations': m,
        'n_gen_samples': m,
        'p': p,
        'T': T,
        'eval_rate': float(eval_rate),
        'denominator': float(denominator),
    }
    
    return results


# 兼容旧接口
evaluate_imputation_fixed_mask = evaluate_baseline_imputation
