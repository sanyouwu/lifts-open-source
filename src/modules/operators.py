import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la
# import CRPS.CRPS as pscore
# 
import torch
from utils import mask_for_test
# from modules.operators import compute_cov
from itertools import accumulate, islice
# import CRPS.CRPS as pscore


def cal_energy_score2(M1, M2, obs, obs_mask):
    ### obs shape: B x p x T
    ### torch tensor M1 shape: B x gen_times x p x T
    ### torch tensor M2 shape: B x gen_times x p x T
    ### M1 == M2 in the shape
    if len(M1) == 3:
        M1 = M1.unsqueeze(1)
    if len(M2) == 3:
        M2 = M2.unsqueeze(1)
    B, p, T = obs.shape
    B, gen_times, p, T = M1.shape
    term1_all = ((M1 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).sum(dim=(-2, -1))
    term2_all = ((M2 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).sum(dim=(-2, -1))
    
    
    obs_lens = obs_mask.sum(dim=(-2, -1))
    # s2_all = ((M1 - M2)**2 * obs_mask.unsqueeze(1)).mean(dim=(-2, -1))
    with torch.no_grad():  # 确保mask操作不参与梯度计算
        expanded_mask = (obs_mask.unsqueeze(1).unsqueeze(1)).float()
        # print("expanded_mask shape: ", expanded_mask.shape)  # shape: (batch, gen_sample_size, gen_sample_size, p, T)
    M_pred1_exp = M1.unsqueeze(2)
    M_pred2_exp = M2.unsqueeze(1)
    squared_diff = (M_pred1_exp - M_pred2_exp) ** 2
    # print("squared_diff shape: ", squared_diff.shape)  # shape: (batch, gen_sample_size, gen_sample_size, p, T)
    # print("expanded_mask shape: ", expanded_mask.shape)  # shape: (batch, gen_sample_size, gen_sample_size, p, T)
    masked_data = squared_diff * expanded_mask  
    
    # print("squared_diff shape: ", squared_diff.shape)  # shape: (batch, gen_sample_size, gen_sample_size, m, m)
    # break
    rmse = torch.sqrt(masked_data.sum(dim=(-2, -1)))  # shape: (batch, m, m)
    # print("rmse shape: ", rmse.shape)  # shape: (batch, gen_sample_size, gen_sample_size)
    # print("(rmse.sum(dim=(1, 2)) shape: ", (rmse.sum(dim=(1, 2))).shape)  # shape: (batch, gen_sample_size)
    energy_loss = (rmse.sum(dim=(1, 2)))/(2 * gen_times * (gen_times-1))
    # print("energy loss:", energy_loss)
    # 合并数学运算
    part1 = torch.sqrt(term1_all).mean(1)
    # part2 = torch.sqrt(term2_all).mean().item()
    # print("part1 :", part1)
    # print("part2 :", part2)
    return ((part1  - energy_loss)/torch.sqrt(obs_lens)).mean().item()
def cal_energy_score(M1, M2, obs, obs_mask):
    ### obs shape: B x p x T
    ### torch tensor M1 shape: B x gen_times x p x T
    ### torch tensor M2 shape: B x gen_times x p x T
    ### M1 == M2 in the shape
    if len(M1) == 3:
        M1 = M1.unsqueeze(1)
    if len(M2) == 3:
        M2 = M2.unsqueeze(1)
    B, p, T = obs.shape
    B, gen_times, p, T = M1.shape
    obs_lens = torch.sqrt(obs_mask.sum(dim=(-2, -1)))
    # energy_score = torch.mean((M1 - M2) ** 2, dim=(1, 2, 3))  # [B]
    # cond_enscore = 0
    # for j in range(1):
    #     term1 = ((M1[:,j,...] - obs)**2 * obs_mask).sum(dim=(-2, -1))
    #     term2 = ((M2[:,j,...] - obs)**2 * obs_mask).sum(dim=(-2, -1))
    #     s2 = ((M1[:,j,...] - M2[:,j])**2 * obs_mask).sum(dim=(-2, -1))
    #     cond_enscore += ((torch.pow(term1, 1/2)).mean() + (torch.pow(term2, 1/2)).mean() - (torch.pow(s2, 1/2)).mean()).item()
    # cond_enscore /= 1
    # return cond_enscore
    # term1_all = ((M1 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).mean(dim=(-2, -1))
    # term2_all = ((M2 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).mean(dim=(-2, -1))
    # s2_all = ((M1 - M2)**2 * obs_mask.unsqueeze(1)).mean(dim=(-2, -1))
    term1_all = ((M1 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).sum(dim=(-2, -1))
    term2_all = ((M2 - obs.unsqueeze(1))**2 * obs_mask.unsqueeze(1)).sum(dim=(-2, -1))
    s2_all = ((M1 - M2)**2 * obs_mask.unsqueeze(1)).sum(dim=(-2, -1))
    # 合并数学运算
    # print("term1_all shape", term1_all.shape)
    part1 = (torch.sqrt(term1_all)).mean(dim=1)/obs_lens
    part2 = (torch.sqrt(term2_all)).mean(dim=1)/obs_lens
    part3 = (torch.sqrt(s2_all)).mean(dim=1)/obs_lens
    
    return (part1 + part2 - part3).mean().item()
    # return part3
    
    
def quantile_loss(target, forecast, q: float, eval_points):
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler = 0, scaler = 1):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def compute_cond_z_stats(X_obs_list, mask_list, Sigma_x, Sigma_z, F_random,M_pred_list,Z_est_lst):
    p = Sigma_x.shape[0]
    n = len(X_obs_list)
    cond_means_list = []
    cond_vars_list = []
    emp_means_list = []
    emp_vars_list = []
    err_mean_list, err_cov_list = [], []
    err_mu_z_lst, err_cov_z_lst = [], []
    device = M_pred_list[0].device 
    Sigma_z = Sigma_z.to(device)  # (p, p)
    Sigma_x = Sigma_x.to(device)  # (p, p)
    for i in range(n):
        # print("i: ", i)
        samples = M_pred_list[i].squeeze()    # (J, p)
        samples_z = Z_est_lst[i].squeeze()    # (J, d)
        x_obs = X_obs_list[i].to(samples.device)  # (p,)
        mask = mask_list[i].to(samples.device)              # (p,)
        x_obs = x_obs.flatten()
        # mask[:4] = False
        
        # print("samples size: ", samples.shape)
        J = samples.shape[0]
        non_zero_count = mask.sum().item() ###2
        # print(f"non_zero_count: {non_zero_count}, J: {J}")
        missing_count = p - non_zero_count
        obs_idx = torch.where(mask)[0]
        mis_idx = torch.where(~mask)[0]
        # print(f"missing count: {missing_count}, obs count: {non_zero_count}")
        # print("obs_idx: ", obs_idx)
        # print("mis_idx: ", mis_idx)
        # break
        if missing_count> 0:
            # print(f"missing count: {missing_count}, obs count: {non_zero_count}")
            ### block-wise F-matrix (follows notations of overleaf)
            F_obs = F_random[obs_idx] ### |obs| x d
            ### block-wise covariance matrix
            Sigma_oo = Sigma_x[obs_idx][:, obs_idx]
            Sigma_mo = Sigma_x[mis_idx][:, obs_idx]
            Sigma_mm = Sigma_x[mis_idx][:, mis_idx]
            # inverse
            # L_oo = torch.linalg.cholesky((Sigma_oo + 1e-6*torch.eye(len(obs_idx))) )
            # inv_Sigma_oo = torch.cholesky_inverse(L_oo)
            inv_Sigma_oo = torch.linalg.pinv(Sigma_oo)
            ##########>>>>>>>>>>>>>> conditional mean and covariance of x
            cond_mu = Sigma_mo @ (inv_Sigma_oo @ x_obs[obs_idx])
            # cond_mu = x_obs[mis_idx]
            cond_cov = Sigma_mm - Sigma_mo @ inv_Sigma_oo @ Sigma_mo.T
            # print(f"conditional mean shape: {cond_mu.shape}, conditional covariance shape: {cond_cov.shape}")
            est_cond_mu = samples.mean(dim=0)[mis_idx]#.cpu()  # (missing_count,)
            est_cond_cov = compute_cov(samples[:, mis_idx])
            err_mean = torch.norm(est_cond_mu - cond_mu) / torch.norm(cond_mu)
            err_cov = la.norm(est_cond_cov - cond_cov.cpu().numpy(),"fro")  / la.norm(cond_cov.cpu().numpy(), "fro")
            
            # err_cov =  la.norm(cond_cov.cpu().numpy(), "fro")
            # print(f"err_mean: {err_mean:.4f}, err_cov: {err_cov:.4f}")
            cond_means_list.append(cond_mu)
            cond_vars_list.append(cond_cov)
            emp_means_list.append(est_cond_mu)
            emp_vars_list.append(est_cond_cov)
            err_mean_list.append(err_mean.item())
            err_cov_list.append(err_cov.item())
            #######--------<<<<<<<<<<<<<<<<<<< conditional mean and covariance of x
            #######-------->>>>>>>>>>>>>>>>>>>>>>>>>>>>conditional mean and covariance of z
            cond_mu_z = Sigma_z @ (F_obs).T @ inv_Sigma_oo @ x_obs[obs_idx]
            cond_cov_z = Sigma_z - Sigma_z @ (F_obs).T @ inv_Sigma_oo @ F_obs @ Sigma_z
            est_cond_mu_z = samples_z.mean(dim=0)
            est_cond_cov_z = (samples_z -est_cond_mu_z).T @ (samples_z - est_cond_mu_z)/ samples_z.shape[0]
            err_mu_z = torch.norm(est_cond_mu_z - cond_mu_z) / torch.norm(cond_mu_z)
            err_cov_z = torch.norm(cond_cov_z - est_cond_cov_z,"fro")/ torch.norm(cond_cov_z,"fro")
            err_mu_z_lst.append(err_mu_z.item())
            err_cov_z_lst.append(err_cov_z.item())
            # print(f"err_mu_z: {err_mu_z:.4f}, err_cov_z: {err_cov_z:.4f}")

    return err_mean_list, err_cov_list, err_mu_z_lst, err_cov_z_lst


# def masked_mae_cal(inputs, target, mask):
#     """calculate Mean Absolute Error"""
#     return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


# def masked_mse_cal(inputs, target, mask):
#     """calculate Mean Square Error"""
#     return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


# def masked_rmse_cal(inputs, target, mask):
#     """calculate Root Mean Square Error"""
#     return torch.sqrt(masked_mse_cal(inputs, target, mask))

def masked_mae_cal(M_pred, target, mask, loss = "mae"):
    """
    calculate Mean Absolute Error
    works for batch_size > 1
    """
    if M_pred.dim() == 3:  # (num_samples, p, t)
        # 扩展维度以适配批处理
        M_pred = M_pred.unsqueeze(0)  # (1, num_samples, p, t)
        target = target.unsqueeze(0).unsqueeze(1)  # (1, 1, p, t)
        mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, p, t)
    elif M_pred.dim() == 4:  # (batch_size, num_samples, p, t)
        # 扩展目标与掩码维度
        target = target.unsqueeze(1)  # (batch_size, 1, p, t)
        mask = mask.unsqueeze(1)  # (batch_size, 1, p, t)
    # mae_cal = 0
    # for gen_sample in M_pred:
    #     mae_cal +=torch.sum(torch.abs(gen_sample - target) * mask) / (torch.sum(mask) + 1e-9)
    # mae_cal /= M_pred.shape[0]
    # return mae_cal
    if loss == "mae":
        abs_error = torch.abs(M_pred - target) 
    else :
        abs_error = (M_pred - target) ** 2  # (batch_size, gen_sample_size, p, T)
    masked_abs_error = abs_error * mask
    numerator = torch.sum(masked_abs_error, dim=(-2,-1))  # sum over (gen_sample_size, p, T)
    denominator = torch.sum(mask, dim=(-2,-1)) + 1e-9  # sum over (gen_sample_size, p, T)
    mae = numerator / denominator  # (batch_size,)
    batch_mae = torch.mean(mae, dim=1)
    return batch_mae  # return the mean over batch_size
def compute_conditional_stats(X_obs_list, mask_list,mask_list_a, Sigma, M_pred_list,mask_test_flag = False, device='cuda:2'):
    p = Sigma.shape[0]
    n = len(X_obs_list)
    
    cond_means_list = []
    cond_vars_list = []
    emp_means_list = []
    emp_vars_list = []
    err_mean_list, err_cov_list = [], []
    device = M_pred_list[0].device if isinstance(M_pred_list[0], torch.Tensor) else device
    Sigma = Sigma.to(device)  # (p, p)
    for i in range(n):
        samples = M_pred_list[i].squeeze()    # (J, p)
        x_obs = X_obs_list[i].to(samples.device)  # (p,)
        mask = mask_list[i].to(samples.device)              # (p,)
        target_mask = mask_list_a[i].to(samples.device)  # (p,)
        # if mask_test_flag:
            # mask = mask_for_test(mask, num_to_select=10)
            # x_obs = x_obs * mask
        x_obs = x_obs.flatten()
        # mask[:4] = False
        
        # print("samples size: ", samples.shape)
        J = samples.shape[0]
        non_zero_count = mask.sum().item()
        missing_count = p - non_zero_count
        obs_idx = torch.where(mask)[0]
        # mis_idx = torch.where(~mask)[0]
        mis_idx = torch.where(target_mask)[0]
        # print(f"missing count: {missing_count}, obs count: {non_zero_count}")
        # print("obs_idx: ", obs_idx)
        # print("mis_idx: ", mis_idx)
        # break
        if missing_count> 0:
            # 分块协方差矩阵
            Sigma_oo = Sigma[obs_idx][:, obs_idx]
            Sigma_mo = Sigma[mis_idx][:, obs_idx]
            Sigma_mm = Sigma[mis_idx][:, mis_idx]
            # inverse
            # L_oo = torch.linalg.cholesky((Sigma_oo + 1e-6*torch.eye(len(obs_idx))) )
            # inv_Sigma_oo = torch.cholesky_inverse(L_oo)
            inv_Sigma_oo = torch.linalg.pinv(Sigma_oo)
            
            # conditional mean and covariance
            # print(x_obs.dtype)
            # print(Sigma_mo.dtype)
            # print(inv_Sigma_oo.dtype)
            # print(x_obs[obs_idx])
            # print(x_obs[obs_idx].dtype)
            cond_mu = Sigma_mo @ (inv_Sigma_oo @ x_obs[obs_idx])
            # cond_mu = x_obs[mis_idx]
            cond_cov = Sigma_mm - Sigma_mo @ inv_Sigma_oo @ Sigma_mo.T
            # print(f"conditional mean shape: {cond_mu.shape}, conditional covariance shape: {cond_cov.shape}")
            est_cond_mu = samples.mean(dim=0)[mis_idx]#.cpu()  # (missing_count,)
            # est_cond_mu = x_obs[mis_idx]
            # print("samples[:, mis_idx] shape: ", samples[:, mis_idx].shape)
            est_cond_cov = compute_cov(samples[:, mis_idx])
            # print(samples.shape)
            # print(f"est_cond_mu shape: {est_cond_mu.shape}, est_cond_cov shape: {est_cond_cov.shape}")
            # print(f"cond_mu shape: {cond_mu.shape}, cond_cov shape: {cond_cov.shape}")
            err_mean = torch.norm(est_cond_mu - cond_mu) / torch.norm(cond_mu)
            err_cov = la.norm(est_cond_cov - cond_cov.cpu().numpy(),"fro")  / la.norm(cond_cov.cpu().numpy(), "fro")
            # err_cov =  la.norm(cond_cov.cpu().numpy(), "fro")
            # print(f"err_mean: {err_mean:.4f}, err_cov: {err_cov:.4f}")
            cond_means_list.append(cond_mu)
            cond_vars_list.append(cond_cov)
            emp_means_list.append(est_cond_mu)
            emp_vars_list.append(est_cond_cov)
            err_mean_list.append(err_mean.item())
            err_cov_list.append(err_cov.item())
    
    return cond_means_list, cond_vars_list, emp_means_list, emp_vars_list,err_mean_list, err_cov_list

def heatmap_annot(data,top_num = 20):
    flat_data = np.abs(data).flatten()
    _len = len(flat_data)
    if top_num >= _len:
        threshold = 0
    else:
        threshold = np.partition(flat_data, -top_num)[-top_num]
    annot = np.where(np.abs(data) > threshold, np.round(data,2), "")
    return annot

def compute_cov(mat, cond_mu = None):
    ### convert tensor to numpy
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy().squeeze()
    if cond_mu is not None:
        if isinstance(cond_mu, torch.Tensor):
            cond_mu = cond_mu.cpu().numpy().squeeze()
        mat = mat - cond_mu
    else:
        mat = mat - np.mean(mat, axis = 0)
    
    return (mat.T @ mat) / mat.shape[0]
def calculate_coverage(a: torch.Tensor, b: torch.Tensor):
    """
    计算b是否落在a的95%预测区间内
    Args:
        a: 预测结果 (n, 100)
        b: 真实值 (n,)
    Returns:
        coverage_mask: 0/1覆盖标记 (n,)
    """
    # 计算分位数（并行化实现）
    quantiles_80 = torch.quantile(
        a,
        # torch.tensor([0.025, 0.975], device=a.device),
        torch.tensor([0.1, 0.9], device=a.device),
        dim=1
    )
    quantiles_95 = torch.quantile(
        a,
        torch.tensor([0.025, 0.975], device=a.device),
        # torch.tensor([0.1, 0.9], device=a.device),
        dim=1
    )
    
    # 获取上下界
    lower_bound_80, upper_bound_80 = quantiles_80[0], quantiles_80[1]
    lower_bound_95, upper_bound_95 = quantiles_95[0], quantiles_95[1]
    # 生成覆盖标记（向量化比较）
    res_80 = ((b >= lower_bound_80) & (b <= upper_bound_80)).to(torch.float32)
    res_95 = ((b >= lower_bound_95) & (b <= upper_bound_95)).to(torch.float32)
    # print("res shape/: ", res.shape)
    # print(res[:5])
    return torch.mean(res_80, dim=0).item(), torch.mean(res_95, dim=0).item()
def calculate_coverage_optimized(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """
    加速版覆盖率计算（合并分位数提取）
    Args:
        a: 预测结果 (n, 100)
        b: 真实值 (n,)
        c: 另一真实值（假设需单独处理） (n,)
    Returns:
        (coverage_60, coverage_80, coverage_95)
    """
    # 一次性计算所有分位数（减少重复排序）
    # q = torch.tensor([0.025, 0.1, 0.2, 0.8, 0.9, 0.975], device=a.device)
    q = torch.tensor([0.05, 0.1, 0.2, 0.8, 0.9, 0.95], device=a.device)
    quantiles = torch.quantile(a, q, dim=1)  # 形状 [6, n]
    
    # 解包分位数（避免多次索引）
    lower_95, lower_80, lower_60, upper_60, upper_80, upper_95 = quantiles
    
    # 向量化比较（无需逐元素循环）
    coverage_60 = (b >= lower_60) & (b <= upper_60)
    coverage_80 = (b >= lower_80) & (b <= upper_80)
    coverage_95 = (c >= lower_95) & (c <= upper_95)
    
    #### the length of coverage_60, coverage_80, coverage_95 is n
    width_60 = upper_60 - lower_60
    width_80 = upper_80 - lower_80
    width_95 = upper_95 - lower_95
    # 直接计算均值（避免中间转CPU）
    return (
        coverage_60.float().mean().item(),
        coverage_80.float().mean().item(),
        coverage_95.float().mean().item(),
        width_60.float().mean().item(),
        width_80.float().mean().item(),
        width_95.float().mean().item()
    )
    
# def crps_missing(real_value,pred_value,mask):
#     crps_normalized = calc_quantile_CRPS(real_value, pred_value, mask, mean_scaler=0, scaler=1)
#     pred_value =  pred_value.permute(0,2,3,1)
#     # real_value = real_value[mask]
#     masked_pred = pred_value[mask]
#     masked_real = real_value[mask]
#     # print("masked_pred shape: ",masked_pred.shape)
#     # print("masked real shape",masked_real.shape)
#     cover_60, cover_80,cover_95,CI_60,CI_80,CI_95 = calculate_coverage_optimized(masked_pred, masked_real,masked_real)
    
#     return crps_normalized,cover_60, cover_80,cover_95,CI_60,CI_80,CI_95
def crps_missing(real_value,pred_value,mask):
    
    pred_value =  pred_value.permute(0,2,3,1)
    # real_value = real_value[mask]
    
    masked_pred = pred_value[mask]
    masked_real = real_value[mask]
    # print("masked_pred shape: ",masked_pred.shape)
    # print("masked real shape",masked_real.shape)
    cover_60, cover_80,cover_95,CI_60,CI_80,CI_95 = calculate_coverage_optimized(masked_pred, masked_real,masked_real)
    masked_pred = masked_pred.contiguous().cpu().numpy()
    masked_real = masked_real.contiguous().cpu().numpy()
    crps_lst = []
    denominator_lst = []
    for i in range(len(masked_real)):
        try:
            crps_single,_,_ = (pscore(masked_pred[i],masked_real[i]).compute())
            crps_lst.append(crps_single)
            denominator_lst.append(masked_real[i])
        except:
            crps_lst.append(0)
            denominator_lst.append(1)
            print("wrong in crps_missing, i: ",i)
            continue
    # 
    cut_list = mask.sum(dim = (-1,-2)).tolist()
    end_points = [0] + list(accumulate(cut_list))
    crps_segs = [list(islice(crps_lst, start, end)) 
               for start, end in zip(end_points, end_points[1:])]
    crps_segs_sum = [np.sum(np.abs(crps_seg)) for crps_seg in crps_segs]
    denominator_segs = [list(islice(denominator_lst, start, end))
               for start, end in zip(end_points, end_points[1:])]
    denominator_segs_sum = [np.sum(np.abs(denominator_seg)) for denominator_seg in denominator_segs]
    # return crps_lst,denominator_lst#,cover_60, cover_80,cover_95,CI_60,CI_80,CI_95
    crps_normalized = (np.array(crps_segs_sum)/np.array(denominator_segs_sum)).tolist()
    return crps_normalized,cover_60, cover_80,cover_95,CI_60,CI_80,CI_95
    # return np.mean(crps_lst)/np.mean(np.abs(denominator_lst)),cover_60, cover_80,cover_95,CI_60,CI_80,CI_95
    # return np.mean(crps_lst),cover_60, cover_80,cover_95,CI_60,CI_80,CI_95
    
# def crps_missing(real_value,pred_value,mask):
#     # mask = mask.squeeze().cpu().numpy()
#     # mask = mask.cpu().numpy()
#     pred_value =  pred_value.permute(0,2,3,1)
#     # real_value = real_value[mask]
    
#     masked_pred = pred_value[mask]
#     masked_real = real_value[mask]
    
#     cover_80,cover_95 = calculate_coverage(masked_pred, masked_real)
#     # print("masked_pred shape: ",masked_pred.shape)
#     # print("masked real shape",masked_real.shape)
#     masked_pred = masked_pred.contiguous().cpu().numpy()
#     masked_real = masked_real.contiguous().cpu().numpy()
#     crps_lst = []
#     denominator_lst = []
#     for i in range(len(real_value)):
#         crps_single,_,_ = (pscore(masked_pred[i],masked_real[i]).compute())
#         crps_lst.append(crps_single)
#         denominator_lst.append(masked_real[i])
#     return np.mean(crps_lst)/np.mean(np.abs(denominator_lst)), cover_80,cover_95
#### original version
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def crps_missing(real_value,pred_value,mask):
#     # mask = mask.squeeze().cpu().numpy()
#     # mask = mask.cpu().numpy()
#     pred_value =  pred_value.permute(0,2,3,1)
#     # real_value = real_value[mask]
    
#     masked_pred = pred_value[mask]
#     masked_real = real_value[mask]
    
#     masked_pred = masked_pred.contiguous().cpu().numpy()
#     masked_real = masked_real.contiguous().cpu().numpy()
#     crps_lst = []
#     for i in range(len(real_value)):
#         crps_single,_,_ = (pscore(masked_pred[i],masked_real[i]).compute())
#         crps_lst.append(crps_single)
#     return np.mean(crps_lst),1

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

# quantiles_imp[2][dataind,0,:]
def max_eig(mat):
    return np.max(np.abs(la.eigvals(mat)))

def gen_sigma(phi, phi_e):
    ### phi_e is the coefficiemt matrix for noise
    temp = 0
    temp2 = phi_e @ (phi_e.T)
    while True:
        temp += temp2
        temp2 = phi @ (temp2) @ (phi.T)
        if np.sum(temp2**2) < 1e-5:
            break
    return temp

