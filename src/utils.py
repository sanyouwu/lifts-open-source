import torch
import torch.nn as nn
import numpy as np
def generate_artificial_mask(Omega, mask_ratio=0.4):
    observed_pos = torch.where(Omega == 1)
    rand_mask = torch.rand_like(Omega) < mask_ratio
    return Omega * (1 - rand_mask)



def normalize(X, Omega):
    mean = X[Omega==1].mean()
    std = X[Omega==1].std()
    return (X - mean)/(std + 1e-6), mean, std


def generate_mask(k: int, p: int) :
    """
    """
    assert p % 2 == 0, "p 必须是偶数"
    
    # 生成随机矩阵并排序
    rand_vals = torch.rand(k, p)
    sorted_indices = rand_vals.argsort(dim=1)
    
    # 创建初始全 True 掩码
    mask = torch.ones(k, p, dtype=torch.bool)
    
    # 设置前 p//2 个最小值为 False
    mask.scatter_(dim=1, index=sorted_indices[:, :p//2], value=False)
    
    return mask
def mask_for_test(observed_masks, num_to_select=20):
    # Ensure observed_masks is a PyTorch tensor
    if not isinstance(observed_masks, torch.Tensor):
        raise ValueError("observed_masks must be a PyTorch tensor")
    masks = observed_masks.clone().view(-1)
    # print("masks shape: ", masks.shape)
    obs_indices = torch.nonzero(masks, as_tuple=True)[0].tolist()
    if len(obs_indices) <= 1:
        print("obs_indices: ", obs_indices)
        print("warning: No indices to select from. Returning original masks.")
    if num_to_select == 0 and len(obs_indices) >= 1:
        # If no indices are selected, select one randomly
        if len(obs_indices) >= 4:
            num_to_select = 2
        else:
            num_to_select = 1
    remain_num = len(obs_indices) - num_to_select
    if remain_num == 0:
        print("warning: No indices for Omega_b.")
    if num_to_select ==0:
        print("warning: No indices for Omega_a.")
    # print(f"len(obs_indices): {len(obs_indices)} ,num_to_select: {num_to_select}, remain_num: {remain_num}")
    miss_indices = [obs_indices[i] for i in np.random.choice(len(obs_indices), num_to_select, replace=False)]
    # miss_indices = [obs_indices[i] for i in range(num_to_select)]
    # print("obs_indices: ", obs_indices)
    # print("miss_indices: ", miss_indices)
    # Set selected indices to False
    masks[miss_indices] = False
    
    # Reshape back to original shape
    gt_masks = masks.view(observed_masks.shape)
    return gt_masks