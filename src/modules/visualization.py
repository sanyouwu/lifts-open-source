import torch
import numpy as np
import matplotlib.pyplot as plt
###############################

def plot_probabilistic_imputation(imputation_samples, ground_truth, p, T, obs_mask=None, title=None):
    """
    绘制概率插补图。
    
    标记：
    1. Observed Points: 红色叉号 ('x')
    2. Missing Points: 黑色圆点 ('o')
    3. Prediction: 蓝色中位数线 + 95% CI
    """
    # 1. 数据预处理 (计算分位数)
    lower_bound = torch.quantile(imputation_samples, 0.025, dim=0).detach().cpu().numpy()
    upper_bound = torch.quantile(imputation_samples, 0.975, dim=0).detach().cpu().numpy()
    median_pred = torch.quantile(imputation_samples, 0.5, dim=0).detach().cpu().numpy()
    
    gt = ground_truth.detach().cpu().numpy()
    
    # 处理 Mask
    if obs_mask is not None:
        if isinstance(obs_mask, torch.Tensor):
            mask = obs_mask.detach().cpu().numpy()
        else:
            mask = obs_mask
        if mask.ndim == 3: mask = mask[0]
    else:
        mask = np.ones_like(gt)  # 默认全观测

    # 2. 设置绘图布局
    num_series = p
    time_steps = np.arange(1, T + 1)
    ncols = min(4, num_series)
    nrows = (num_series + ncols - 1) // ncols
    figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if num_series > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 3. 循环绘制
    for i in range(num_series):
        ax = axes[i]
        
        # A. 蓝色置信区间 (95% CI)
        ax.fill_between(
            time_steps, 
            lower_bound[i], 
            upper_bound[i], 
            color='blue', 
            alpha=0.2, 
            label='95% CI'
        )
        
        # B. 蓝色中位数线
        ax.plot(
            time_steps, 
            median_pred[i], 
            color='blue', 
            linewidth=1, 
            linestyle='-',
            alpha=0.6,
            label='Median Pred'
        )
        
        # C. 标记缺失点 (黑色圆点 'o')
        missing_idx = np.where(mask[i] == 0)[0]
        if len(missing_idx) > 0:
            ax.scatter(
                time_steps[missing_idx], 
                gt[i][missing_idx], 
                color='black',
                marker='o',
                s=30,
                zorder=5,
                label='Missing'
            )
        
        # D. 标记观测点 (红色叉号 'x')
        observed_idx = np.where(mask[i] == 1)[0]
        if len(observed_idx) > 0:
            ax.scatter(
                time_steps[observed_idx], 
                gt[i][observed_idx], 
                color='red',
                marker='x',
                s=40,
                linewidth=1.5,
                zorder=6,
                label='Observed'
            )

        # E. 美化
        ax.set_title(f'Feature {i+1}', fontsize=10, fontweight='bold')
        ax.set_xlim(0.5, T + 0.5)
        ax.set_xticks(np.arange(2, T+1, 2))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 图例 (去重)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

    fig_title = title if title else 'Probabilistic Prediction (Red X=Observed, Black O=Missing)'
    plt.suptitle(fig_title, fontsize=14)
    plt.show()


def plot_probabilistic_forecast(forecast_samples, ground_truth, p, T, obs_mask=None, title=None):
    """
    绘制概率预测图。
    
    标记：
    1. Observed Points: 红色叉号 ('x')
    2. Missing Points: 黑色圆点 ('o')
    3. Prediction: 蓝色中位数线 + 95% CI
    """
    
    # 1. 数据预处理 (计算分位数)
    lower_bound = torch.quantile(forecast_samples, 0.025, dim=0).detach().cpu().numpy()
    upper_bound = torch.quantile(forecast_samples, 0.975, dim=0).detach().cpu().numpy()
    median_pred = torch.quantile(forecast_samples, 0.5, dim=0).detach().cpu().numpy()
    
    gt = ground_truth.detach().cpu().numpy()
    
    # 处理 Mask
    if obs_mask is not None:
        if isinstance(obs_mask, torch.Tensor):
            mask = obs_mask.detach().cpu().numpy()
        else:
            mask = obs_mask
        if mask.ndim == 3: mask = mask[0]
    else:
        mask = np.zeros_like(gt)

    # 2. 设置绘图布局
    num_series = p
    time_steps = np.arange(1, T + 1)
    ncols = min(4, num_series)
    nrows = (num_series + ncols - 1) // ncols
    figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if num_series > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 3. 循环绘制
    for i in range(num_series):
        ax = axes[i]
        
        # A. 蓝色置信区间 (95% CI)
        ax.fill_between(
            time_steps, 
            lower_bound[i], 
            upper_bound[i], 
            color='blue', 
            alpha=0.2, 
            label='95% CI'
        )
        
        # B. 蓝色中位数线
        ax.plot(
            time_steps, 
            median_pred[i], 
            color='blue', 
            linewidth=1, 
            linestyle='-',
            alpha=0.6,
            label='Median Pred'
        )
        
        # C. 标记缺失点 (黑色圆点 'o')
        missing_idx = np.where(mask[i] == 0)[0]
        if len(missing_idx) > 0:
            ax.scatter(
                time_steps[missing_idx], 
                gt[i][missing_idx], 
                color='black',
                marker='o',
                s=30,
                zorder=5,
                label='Missing'
            )
        
        # D. 标记观测点 (红色叉号 'x')
        observed_idx = np.where(mask[i] == 1)[0]
        if len(observed_idx) > 0:
            ax.scatter(
                time_steps[observed_idx], 
                gt[i][observed_idx], 
                color='red',
                marker='x',
                s=40,
                linewidth=1.5,
                zorder=6,
                label='Observed'
            )

        # E. 美化
        ax.set_title(f'Feature {i+1}', fontsize=10, fontweight='bold')
        ax.set_xlim(0.5, T + 0.5)
        ax.set_xticks(np.arange(2, T+1, 2))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 图例 (去重)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

    fig_title = title if title else 'Probabilistic Prediction (Red X=Observed, Black O=Missing)'
    plt.suptitle(fig_title, fontsize=14)
    plt.show()


# def plot_probabilistic_forecast(forecast_samples, ground_truth, p, T, obs_mask=None):
#     """
#     绘制概率预测图：包含 95% 置信区间、中位数预测和真实值。
    
#     Args:
#         forecast_samples: Tensor shape (100, 10, 20) -> (Samples, Series, Time)
#         ground_truth: Tensor shape (10, 20) -> (Series, Time) 真实值矩阵
#                       (如果真实值中有 NaN，matplotlib 会自动断开线条，这符合预期)
#     """
    
#     # 1. 数据预处理：计算分位数 (Quantiles)
#     # 在 dim=0 (样本维度 100) 上计算
#     # lower: 2.5% 分位数
#     # upper: 97.5% 分位数
#     # median: 50% 分位数 (作为预测的主趋势线)
#     lower_bound = torch.quantile(forecast_samples, 0.1, dim=0).detach().cpu().numpy()
#     upper_bound = torch.quantile(forecast_samples, 0.9, dim=0).detach().cpu().numpy()
#     median_pred = torch.quantile(forecast_samples, 0.5, dim=0).detach().cpu().numpy()
    
#     # 转换真实值为 numpy
#     gt = ground_truth.detach().cpu().numpy()
# # ---------------------------------------------------------
#     # 核心修改：应用 Mask 强制覆盖
#     # ---------------------------------------------------------
#     if obs_mask is not None:
#         if isinstance(obs_mask, torch.Tensor):
#             mask = obs_mask.detach().cpu().numpy()
#         else:
#             mask = obs_mask
            
#         # 确保 mask 是 (p, T)
#         if mask.ndim == 3: mask = mask[0] 
        
#         # 逻辑：
#         # 在 mask=1 的地方，上下界和中位数都强制等于 ground_truth
#         # 在 mask=0 的地方，保持模型原本的预测
#         # 公式: new_val = old_val * (1 - mask) + gt * mask
        
#         lower_bound = lower_bound * (1 - mask) + gt * mask
#         upper_bound = upper_bound * (1 - mask) + gt * mask
#         median_pred = median_pred * (1 - mask) + gt * mask
#     else:
#         # 如果没传 mask，默认全为 0 (全都没观测到，纯预测)
#         mask = np.zeros_like(gt)
#     # 2. 设置绘图布局 (5行2列 = 10个子图)
#     num_series = p
#     time_steps = np.arange(1, T + 1) # 横坐标 1 到 T
#     ncols = min(4,num_series)
#     # nrows = p//4 +1 if p %4 !=0 else p//4
#     nrows = (num_series + ncols - 1) // ncols
#     figsize = (4 * ncols, 3 * nrows)
    
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
#     axes = axes.flatten() # 展平数组方便遍历
    
#     # 3. 循环绘制每一个子图
#     for i in range(num_series):
#         ax = axes[i]
        
#         # A. 绘制 95% 置信区间 (Confidence Interval)
#         # alpha 设置透明度，让背景网格可见
#         ax.fill_between(
#             time_steps, 
#             lower_bound[i], 
#             upper_bound[i], 
#             color='blue', 
#             alpha=0.2, 
#             label='95% Confidence Interval'
#         )
        
#         # B. 绘制预测的中位数 (Median Prediction)
#         ax.plot(
#             time_steps, 
#             median_pred[i], 
#             color='blue', 
#             linewidth=1.5, 
#             linestyle='-', 
#             label='Median Prediction'
#         )
        
#         # C. 绘制真实值 (Ground Truth)
#         # 使用 'o--' (圆点+虚线) 来清晰展示离散的时间点
#         # color='k' (黑色) 形成鲜明对比
#         ax.plot(
#             time_steps, 
#             gt[i], 
#             'o--', 
#             color='black', 
#             markersize=5, 
#             label='Ground Truth'
#         )
        
#         # D. 设置坐标轴和标题
#         ax.set_title(f'Time Series Feature {i+1}', fontsize=10, fontweight='bold')
#         ax.set_xlim(1, 20)
#         ax.set_xticks(np.arange(2, 21, 2)) # 每2个步长显示刻度
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # 只在第一个图显示图例，避免杂乱
#         if i == 0:
#             ax.legend(loc='upper left', fontsize=8)

#     # 设置整体标题
#     plt.suptitle('Probabilistic Forecasting (95% CI vs Ground Truth)', fontsize=16)
#     plt.show()


########################
# def compare_distribution_overlap(my_samples, kalman_samples, title_prefix="Feature"):
#     """
#     绘制两个分布随时间变化的重合程度对比图。
    
#     Args:
#         my_samples (Tensor): (gen_size, p, T) - 你的模型生成的样本
#         kalman_samples (Tensor): (gen_size, p, T) - Kalman Filter 生成的理论样本
#         title_prefix (str): 子图标题前缀
#     """
#     # 1. 获取维度信息
#     # 假设形状是 (Sample_Size, Feature_Num, Time_Steps)
#     _, num_features, time_steps = my_samples.shape
    
#     # 将 Tensor 转为 CPU numpy，方便绘图
#     my_samples = my_samples.detach().cpu()
#     kalman_samples = kalman_samples.detach().cpu()
    
#     # 2. 计算统计量 (Quantiles)
#     # 我们需要 2.5% (下界), 50% (中位数/均值), 97.5% (上界)
#     # dim=0 表示沿着 sample_size 维度计算
    
#     # --- My Model Stats ---
#     my_lower = torch.quantile(my_samples, 0.025, dim=0).numpy()
#     my_upper = torch.quantile(my_samples, 0.975, dim=0).numpy()
#     my_mean  = torch.quantile(my_samples, 0.5, dim=0).numpy() # 用中位数代表趋势
    
#     # --- Kalman Stats ---
#     kf_lower = torch.quantile(kalman_samples, 0.025, dim=0).numpy()
#     kf_upper = torch.quantile(kalman_samples, 0.975, dim=0).numpy()
#     kf_mean  = torch.quantile(kalman_samples, 0.5, dim=0).numpy()
    
#     # 3. 设置绘图布局
#     # 自动计算行数和列数 (默认 2 列)
#     n_cols = 2
#     n_rows = (num_features + 1) // 2
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), constrained_layout=True)
#     axes = axes.flatten() # 展平方便遍历
    
#     t_axis = np.arange(1, time_steps + 1) # 横坐标 1 -> T
    
#     # 4. 循环绘制每个 Feature
#     for i in range(num_features):
#         ax = axes[i]
        
#         # --- 绘制 Kalman Distribution (基准 - 灰色) ---
#         # Fill: 95% 置信区间
#         ax.fill_between(
#             t_axis, kf_lower[i], kf_upper[i],
#             color='gray', alpha=0.3, 
#             label='Kalman (Theoretical) 95% CI'
#         )
#         # Line: 趋势线
#         ax.plot(
#             t_axis, kf_mean[i], 
#             color='black', linestyle='--', linewidth=1.5, alpha=0.7,
#             label='Kalman Mean'
#         )
        
#         # --- 绘制 My Model Distribution (生成 - 蓝色) ---
#         # Fill: 95% 置信区间
#         ax.fill_between(
#             t_axis, my_lower[i], my_upper[i],
#             color='blue', alpha=0.3, 
#             label='Generated (Model) 95% CI'
#         )
#         # Line: 趋势线
#         ax.plot(
#             t_axis, my_mean[i], 
#             color='blue', linestyle='-', linewidth=1.5, 
#             label='Generated Mean'
#         )
        
#         # --- 装饰 ---
#         ax.set_title(f'{title_prefix} {i+1}', fontsize=10, fontweight='bold')
#         ax.set_xlim(1, time_steps)
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # 为了防止图例遮挡，只在第一个图显示图例
#         if i == 0:
#             ax.legend(loc='best', fontsize=8)

#     # 隐藏多余的子图 (如果 feature 数量是奇数)
#     for j in range(num_features, len(axes)):
#         axes[j].axis('off')
        
#     plt.suptitle(f'Distribution Comparison: Generated vs. Theoretical (Kalman)', fontsize=16)
#     plt.show()
    
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # 引入这个用于控制刻度
import numpy as np

def compare_distribution_overlap(my_samples, kalman_samples, title_prefix="Feature",name1="Generated",name2="Kalman"):
    """
    绘制两个分布随时间变化的重合程度对比图。
    修改点：横坐标强制为整数，显示所有刻度，但每隔5个显示标签。
    """
    # 1. 获取维度信息
    # 假设形状是 (Sample_Size, Feature_Num, Time_Steps)
    _, num_features, time_steps = my_samples.shape
    
    # 将 Tensor 转为 CPU numpy，方便绘图
    my_samples = my_samples.detach().cpu()
    kalman_samples = kalman_samples.detach().cpu()
    
    # 2. 计算统计量
    my_lower = torch.quantile(my_samples, 0.1, dim=0).numpy()
    my_upper = torch.quantile(my_samples, 0.9, dim=0).numpy()
    my_mean  = torch.quantile(my_samples, 0.5, dim=0).numpy()
    
    kf_lower = torch.quantile(kalman_samples, 0.1, dim=0).numpy()
    kf_upper = torch.quantile(kalman_samples, 0.9, dim=0).numpy()
    kf_mean  = torch.quantile(kalman_samples, 0.5, dim=0).numpy()
    
    # 3. 设置绘图布局
    n_cols = 2
    n_rows = (num_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), constrained_layout=True)
    axes = axes.flatten()
    
    t_axis = np.arange(1, time_steps + 1) # 横坐标 1 -> T
    
    # 4. 循环绘制每个 Feature
    for i in range(num_features):
        ax = axes[i]
        
        # --- 绘制 Kalman Distribution (基准 - 灰色) ---
        ax.fill_between(t_axis, kf_lower[i], kf_upper[i], color='gray', alpha=0.3, label=name2 + ' 95% CI')
        ax.plot(t_axis, kf_mean[i], color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=name2 + ' Mean')
        
        # --- 绘制 My Model Distribution (生成 - 蓝色) ---
        ax.fill_between(t_axis, my_lower[i], my_upper[i], color='blue', alpha=0.3, label=name1 + ' 95% CI')
        ax.plot(t_axis, my_mean[i], color='blue', linestyle='-', linewidth=1.5, label=name1 + ' Mean')
        
        # --- 装饰 ---
        ax.set_title(f'{title_prefix} {i+1}', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # =======================================================
        # --- 坐标轴设置 (关键修改) ---
        # =======================================================
        
        # 1. 设置 X 轴范围: 严格限制在 1 到 T
        ax.set_xlim(1, time_steps)
        
        # 2. 主刻度 (Major Ticks): 用于显示标签 (每隔 5 个)
        # 使用 MultipleLocator(5) 让主刻度出现在 5, 10, 15, 20...
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        
        # 3. 次刻度 (Minor Ticks): 用于显示每一个整数位置的刻度线
        # 使用 MultipleLocator(1) 让每 1 个单位都有一个小刻度
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        # 4. 格式化: 强制标签不显示小数 (显示整数 %d)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        # (可选) 如果你想让两边的刻度更明显，可以开启这个
        ax.tick_params(which='both', direction='in') 
        # =======================================================

        if i == 0:
            ax.legend(loc='best', fontsize=8)

    for j in range(num_features, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f'Distribution Comparison: {name1} vs. {name2}', fontsize=16)
    plt.show()
    
    
####### 2d dot plot
import torch
import matplotlib.pyplot as plt
import numpy as np

# def plot_2d_distribution_evolution(my_samples, kalman_samples):
#     """
#     针对 p=2 的情况，绘制每个时间步的 2D 散点分布对比图。
    
#     Args:
#         my_samples (Tensor): (gen_size, 2, T)
#         kalman_samples (Tensor): (gen_size, 2, T)
#     """
#     # 1. 数据准备
#     gen_size, p, T = my_samples.shape
#     assert p == 2, "此函数仅适用于 p=2 的情况 (2D Scatter)"
    
#     # 转为 Numpy
#     my_samples = my_samples.detach().cpu().numpy()
#     kalman_samples = kalman_samples.detach().cpu().numpy()
    
#     # 2. 计算均值 (用于画重心)
#     # Shape: (2, T)
#     my_means = np.mean(my_samples, axis=0)
#     kf_means = np.mean(kalman_samples, axis=0)
    
#     # 3. 设置布局 (自适应网格)
#     # 比如 T=20 -> 4行5列
#     n_cols = 5
#     n_rows = (T + n_cols - 1) // n_cols
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
#     axes = axes.flatten()
    
#     # 4. 循环时间步 T
#     for t in range(T):
#         ax = axes[t]
        
#         # --- A. 绘制 Kalman (基准 - 灰色) ---
#         # 取出第 t 时刻所有样本: x坐标=Feature0, y坐标=Feature1
#         kf_x = kalman_samples[:, 0, t]
#         kf_y = kalman_samples[:, 1, t]
        
#         ax.scatter(kf_x, kf_y, 
#                    c='gray', marker='o', alpha=0.4, s=30, edgecolors='white',
#                    label='Kalman Samples')
        
#         # 画 Kalman 重心 (黑色大星)
#         ax.scatter(kf_means[0, t], kf_means[1, t], 
#                    c='black', marker='*', s=200, edgecolors='white', linewidth=1.5,
#                    label='Kalman Mean', zorder=10)
        
#         # --- B. 绘制 My Model (生成 - 红色) ---
#         my_x = my_samples[:, 0, t]
#         my_y = my_samples[:, 1, t]
        
#         ax.scatter(my_x, my_y, 
#                    c='red', marker='x', alpha=0.5, s=30,
#                    label='Model Samples')
        
#         # 画 Model 重心 (深红大星)
#         ax.scatter(my_means[0, t], my_means[1, t], 
#                    c='darkred', marker='*', s=200, edgecolors='white', linewidth=1.5,
#                    label='Model Mean', zorder=11)
        
#         # --- C. 装饰 ---
#         ax.set_title(f'Time Step {t+1}', fontsize=12, fontweight='bold')
#         ax.set_xlabel('Feature 1 ($x_1$)')
#         ax.set_ylabel('Feature 2 ($x_2$)')
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # 只有第一个图显示图例
#         if t == 0:
#             ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

#     # 隐藏多余的子图
#     for j in range(T, len(axes)):
#         axes[j].axis('off')
        
#     plt.suptitle('2D Distribution Evolution: Model (Red) vs. Kalman (Gray)', fontsize=18)
#     plt.show()

# # ==========================================
# # 测试代码
# # ==========================================
# # if __name__ == "__main__":
# #     # 模拟 p=2 的数据
# #     T = 10
# #     gen_size = 100
    
# #     # 构造两个略有不同的 2D 高斯云团
# #     # Kalman: 均值在 (0,0) 附近移动
# #     kf_data = torch.randn(gen_size, 2, T) 
# #     # 让它随时间飘移一点
# #     kf_data += torch.linspace(0, 2, T).view(1, 1, T)
    
# #     # Model: 模拟一点偏差，比如做了旋转或者方差变大
# #     model_data = kf_data + torch.randn(gen_size, 2, T) * 0.3 # 加点噪音
# #     model_data[:, 0, :] += 0.5 # x 轴偏一点
    
# #     plot_2d_distribution_evolution(model_data, kf_data)

import torch
import matplotlib.pyplot as plt
import numpy as np


# def plot_2d_distribution_evolution(my_samples, kalman_samples, cols2plots = 5):
#     """
#     针对 p=2 的情况，绘制 2D 散点分布对比图。
#     【关键修改】固定坐标轴范围 (Global Axis Limits)，以便观察移动和旋转。
#     """
#     gen_size, p, T = my_samples.shape
#     assert p == 2, "此函数仅适用于 p=2 的情况"
    
#     my_samples = my_samples.detach().cpu().numpy()
#     kalman_samples = kalman_samples.detach().cpu().numpy()
    
#     my_means = np.mean(my_samples, axis=0)
#     kf_means = np.mean(kalman_samples, axis=0)
    
#     # -----------------------------------------------------------
#     # 1. 计算全局坐标范围 (Global Limits)
#     # -----------------------------------------------------------
#     # 把所有数据拼在一起找最大最小值
#     all_data = np.concatenate([my_samples, kalman_samples], axis=0) 
#     # all_data shape: (2*gen_size, 2, T)
    
#     # 找到所有时间步、所有样本中的最小值和最大值
#     x_min = np.min(all_data[:, 0, :])
#     x_max = np.max(all_data[:, 0, :])
#     y_min = np.min(all_data[:, 1, :])
#     y_max = np.max(all_data[:, 1, :])
    
#     # 给一点边距 (Padding)
#     padding = 1.0
#     xlims = (x_min - padding, x_max + padding)
#     ylims = (y_min - padding, y_max + padding)
    
#     # -----------------------------------------------------------
#     # 2. 绘图
#     # -----------------------------------------------------------
#     # cols2plots = 8
#     if T <= cols2plots:
#         selected_indices = np.arange(T)
#         n_cols = T
#     else:
#         selected_indices = np.linspace(0, T-1, cols2plots, dtype=int)
#         n_cols = cols2plots
    
#     fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), constrained_layout=True)
    
#     if n_cols == 1: axes = np.array([axes])
#     else: axes = axes.flatten()
    
#     for i, t in enumerate(selected_indices):
#         ax = axes[i]
        
#         # 画散点
#         kf_x, kf_y = kalman_samples[:, 0, t], kalman_samples[:, 1, t]
#         my_x, my_y = my_samples[:, 0, t], my_samples[:, 1, t]
        
#         ax.scatter(kf_x, kf_y, c='gray', marker='o', alpha=0.3, s=20, label='Kalman')
#         ax.scatter(kf_means[0, t], kf_means[1, t], c='black', marker='*', s=150, label='Kalman Mean')
        
#         ax.scatter(my_x, my_y, c='red', marker='x', alpha=0.4, s=20, label='Model')
#         ax.scatter(my_means[0, t], my_means[1, t], c='darkred', marker='*', s=150, label='Model Mean')
        
#         # ---【关键】应用全局坐标轴范围 ---
#         ax.set_xlim(xlims)
#         ax.set_ylim(ylims)
        
#         # 装饰
#         ax.set_title(f'Time Step {t+1}', fontsize=12, fontweight='bold')
#         ax.set_xlabel('$x_1$')
#         if i == 0:
#             ax.set_ylabel('$x_2$')
#             ax.legend(loc='upper right', fontsize=18) # 图例放右上角，防止遮挡
#         ax.grid(True, linestyle=':', alpha=0.5)
        
#         # # 在图中标注一下均值位置，辅助观察
#         # ax.text(0.05, 0.05, f"Kalman mean: ({kf_means[0,t]:.1f}, {kf_means[1,t]:.1f})", 
#         #         transform=ax.transAxes, fontsize=12, color='black')
#         # ax.text(0.05, 0.15, f"Ours mean: ({my_means[0,t]:.1f}, {my_means[1,t]:.1f})", 
#         #         transform=ax.transAxes, fontsize=12, color='darkred')
        

#     plt.suptitle('Temporal Evolution of Joint Distribution', fontsize=16)
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np

# def plot_2d_distribution_evolution(my_samples, kalman_samples, cols2plots=5):
#     """
#     绘制高颜值的学术风格 2D 分布演化图。
#     改进点：共享 Y 轴、去边框、统一图例、Latex 风格字体。
#     """
#     # --- 0. 设置学术风格参数 (可选，模拟 LaTeX 风格) ---
#     plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams['mathtext.fontset'] = 'cm' # 数学公式使用 Computer Modern 字体
#     plt.rcParams['axes.linewidth'] = 0.8 # 边框线宽

#     gen_size, p, T = my_samples.shape
#     assert p == 2, "此函数仅适用于 p=2 的情况"
    
#     my_samples = my_samples.detach().cpu().numpy()
#     kalman_samples = kalman_samples.detach().cpu().numpy()
    
#     my_means = np.mean(my_samples, axis=0)
#     kf_means = np.mean(kalman_samples, axis=0)
    
#     # --- 1. 计算全局坐标范围 ---
#     all_data = np.concatenate([my_samples, kalman_samples], axis=0) 
#     x_min, x_max = np.min(all_data[:, 0, :]), np.max(all_data[:, 0, :])
#     y_min, y_max = np.min(all_data[:, 1, :]), np.max(all_data[:, 1, :])
    
#     padding = (x_max - x_min) * 0.15 # 稍微留宽一点边距
#     xlims = (x_min - padding, x_max + padding)
#     ylims = (y_min - padding, y_max + padding)
    
#     # --- 2. 确定要画的时间步 ---
#     if T <= cols2plots:
#         selected_indices = np.arange(T)
#         n_cols = T
#     else:
#         selected_indices = np.linspace(0, T-1, cols2plots, dtype=int)
#         n_cols = cols2plots
    
#     # --- 3. 创建画布 ---
#     # sharey=True: 关键设置！所有子图共享 Y 轴，只有第一个显示刻度
#     # sharex=True: 共享 X 轴范围（虽然不共享刻度显示）
#     fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4), 
#                              sharey=True, sharex=False, 
#                              constrained_layout=True)
    
#     if n_cols == 1: axes = np.array([axes])
#     else: axes = axes.flatten()
    
#     # 用于收集图例句柄
#     handles = []
#     labels = []
    
#     for i, t in enumerate(selected_indices):
#         ax = axes[i]
        
#         # --- A. 绘图 ---
#         # Kalman (Gray)
#         s1 = ax.scatter(kalman_samples[:, 0, t], kalman_samples[:, 1, t], 
#                    c='#7f7f7f', marker='o', alpha=0.3, s=15, linewidths=0, # 无边框圆点更柔和
#                    label='Kalman Filter')
#         # Kalman Mean
#         s2 = ax.scatter(kf_means[0, t], kf_means[1, t], 
#                    c='black', marker='*', s=120, edgecolors='white', linewidth=0.8,
#                    label='Kalman Mean', zorder=10)
        
#         # Model (Red) - 使用稍微深一点的红色看起来更稳重
#         s3 = ax.scatter(my_samples[:, 0, t], my_samples[:, 1, t], 
#                    c='#d62728', marker='x', alpha=0.5, s=20, linewidths=1.0,
#                    label='FBI-SSM (Ours)')
#         # Model Mean
#         s4 = ax.scatter(my_means[0, t], my_means[1, t], 
#                    c='#8b0000', marker='^', s=100, edgecolors='white', linewidth=0.8,
#                    label='Model Mean', zorder=11) # 改用三角形区分度更高，或者保持五角星
        
#         # 收集图例 (只在第一次循环收集)
#         if i == 0:
#             handles.extend([s1, s3, s2, s4])
#             labels.extend(['Kalman Filter', 'FBI-SSM (Ours)', 'Kalman Mean', 'Model Mean'])

#         # --- B. 美化坐标轴 (Academic Look) ---
#         ax.set_xlim(xlims)
#         ax.set_ylim(ylims)
        
#         # 标题
#         ax.set_title(f'Time Step $t={t+1}$', fontsize=14, fontweight='medium')
        
#         # 坐标轴标签 (X轴每个都标，Y轴只标第一个)
#         ax.set_xlabel('$x_1$', fontsize=12)
#         if i == 0:
#             ax.set_ylabel('$x_2$', fontsize=12)
        
#         # 去除上方和右方的边框 (Spines)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
        
#         # 让左方和下方的边框稍微淡一点
#         ax.spines['left'].set_alpha(0.6)
#         ax.spines['bottom'].set_alpha(0.6)
        
#         # 网格线更淡雅
#         ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
        
#         # 刻度朝内还是朝外？学术通常默认，或者去掉次要刻度
#         ax.tick_params(axis='both', which='both', length=3, width=0.5, labelsize=10, color='gray')

#     # --- 4. 全局图例 (Legend) ---
#     # 放在图形上方，水平排列
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
#                ncol=4, frameon=False, fontsize=12)
    
#     # 调整布局给图例留位置
#     # plt.subplots_adjust(bottom=0.2) # 如果不使用 constrained_layout
    
#     # 主标题 (可选，Paper里通常写在Caption里，这里可以去掉或者放小一点)
#     # plt.suptitle('Temporal Evolution of Joint Probability Distributions', y=1.05, fontsize=16)
    
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np

# def plot_2d_distribution_evolution(my_samples, kalman_samples, cols2plots=5):
#     """
#     绘制高颜值的学术风格 2D 分布演化图。
#     改进点：给图例加上显眼的边框。
#     """
#     # --- 0. 设置学术风格参数 ---
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams['mathtext.fontset'] = 'cm'
#     plt.rcParams['axes.linewidth'] = 0.8

#     gen_size, p, T = my_samples.shape
#     assert p == 2, "此函数仅适用于 p=2 的情况"
    
#     my_samples = my_samples.detach().cpu().numpy()
#     kalman_samples = kalman_samples.detach().cpu().numpy()
    
#     my_means = np.mean(my_samples, axis=0)
#     kf_means = np.mean(kalman_samples, axis=0)
    
#     # --- 1. 计算全局坐标范围 ---
#     all_data = np.concatenate([my_samples, kalman_samples], axis=0) 
#     x_min, x_max = np.min(all_data[:, 0, :]), np.max(all_data[:, 0, :])
#     y_min, y_max = np.min(all_data[:, 1, :]), np.max(all_data[:, 1, :])
    
#     padding = (x_max - x_min) * 0.15
#     xlims = (x_min - padding, x_max + padding)
#     ylims = (y_min - padding, y_max + padding)
    
#     # --- 2. 确定要画的时间步 ---
#     if T <= cols2plots:
#         selected_indices = np.arange(T)
#         n_cols = T
#     else:
#         selected_indices = np.linspace(0, T-1, cols2plots, dtype=int)
#         n_cols = cols2plots
    
#     # --- 3. 创建画布 ---
#     fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4), 
#                              sharey=True, sharex=False, 
#                              constrained_layout=True)
    
#     if n_cols == 1: axes = np.array([axes])
#     else: axes = axes.flatten()
    
#     handles = []
#     labels = []
    
#     for i, t in enumerate(selected_indices):
#         ax = axes[i]
        
#         # --- A. 绘图 ---
#         s1 = ax.scatter(kalman_samples[:, 0, t], kalman_samples[:, 1, t], 
#                    c='#7f7f7f', marker='o', alpha=0.3, s=15, linewidths=0, 
#                    label='Kalman Filter')
#         s2 = ax.scatter(kf_means[0, t], kf_means[1, t], 
#                    c='black', marker='*', s=120, edgecolors='white', linewidth=0.8,
#                    label='Kalman Mean', zorder=10)
        
#         s3 = ax.scatter(my_samples[:, 0, t], my_samples[:, 1, t], 
#                    c='#d62728', marker='x', alpha=0.5, s=20, linewidths=1.0,
#                    label='FBI-SSM (Ours)')
#         s4 = ax.scatter(my_means[0, t], my_means[1, t], 
#                    c='#8b0000', marker='^', s=100, edgecolors='white', linewidth=0.8,
#                    label='Model Mean', zorder=11)
        
#         if i == 0:
#             handles.extend([s1, s3, s2, s4])
#             labels.extend(['Kalman Filter', 'FBI-SSM (Ours)', 'Kalman Mean', 'Model Mean'])

#         # --- B. 美化坐标轴 ---
#         ax.set_xlim(xlims)
#         ax.set_ylim(ylims)
#         ax.set_title(f'Time Step $t={t+1}$', fontsize=14, fontweight='medium')
#         ax.set_xlabel('$x_1$', fontsize=12)
#         if i == 0:
#             ax.set_ylabel('$x_2$', fontsize=12)
        
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_alpha(0.6)
#         ax.spines['bottom'].set_alpha(0.6)
#         ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
#         ax.tick_params(axis='both', which='both', length=3, width=0.5, labelsize=10, color='gray')

# # --- 4. 全局图例 (Legend) - 终极美化版 ---
#     legend = fig.legend(handles, labels, 
#             # 位置控制
#             loc='upper center',          
#             bbox_to_anchor=(0.5, 0.08),  # 放在底部居中
            
#             # 布局控制
#             ncol=4,                      # 4列 (如果觉得太宽，可以改成 2)
#             columnspacing=2.5,           # 列与列之间的距离 (改大，不拥挤)
#             handletextpad=0.5,           # 符号和文字之间的距离
            
#             # 符号控制 (关键修改!)
#             fontsize=13,                 # 文字大小
#             markerscale=2.5,             # ❗符号放大 2.5 倍❗
#             scatterpoints=1,             # 每个图例只显示 1 个点 (默认可能是3个，很难看)
            
#             # 边框样式
#             frameon=True,                
#             fancybox=False,              # 直角边框
#             edgecolor='black',           
#             facecolor='white',           
#             framealpha=1.0,              
#             borderpad=0.8                # 边框内部留白
#             )
    
#     # 稍微加粗边框线
#     legend.get_frame().set_linewidth(1.0)

#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# def plot_2d_distribution_evolution(my_samples, kalman_samples, cols2plots=5):
#     """
#     绘制高颜值的学术风格 2D 分布演化图 (蓝色系)。
#     改进点：颜色改为蓝色，与时序预测图保持视觉一致。
#     """
#     # 设置字体
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams['mathtext.fontset'] = 'cm'
#     plt.rcParams['axes.linewidth'] = 0.8

#     gen_size, p, T = my_samples.shape
#     assert p == 2, "此函数仅适用于 p=2 的情况"
    
#     my_samples = my_samples.detach().cpu().numpy()
#     kalman_samples = kalman_samples.detach().cpu().numpy()
    
#     my_means = np.mean(my_samples, axis=0)
#     kf_means = np.mean(kalman_samples, axis=0)
    
#     # 计算坐标范围
#     all_data = np.concatenate([my_samples, kalman_samples], axis=0) 
#     x_min, x_max = np.min(all_data[:, 0, :]), np.max(all_data[:, 0, :])
#     y_min, y_max = np.min(all_data[:, 1, :]), np.max(all_data[:, 1, :])
    
#     padding = (x_max - x_min) * 0.15
#     xlims = (x_min - padding, x_max + padding)
#     ylims = (y_min - padding, y_max + padding)
    
#     if T <= cols2plots:
#         selected_indices = np.arange(T)
#         n_cols = T
#     else:
#         selected_indices = np.linspace(0, T-1, cols2plots, dtype=int)
#         n_cols = cols2plots
    
#     fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4), 
#                              sharey=True, sharex=False, 
#                              constrained_layout=True)
    
#     if n_cols == 1: axes = np.array([axes])
#     else: axes = axes.flatten()
    
#     handles = []
#     labels = []
    
#     for i, t in enumerate(selected_indices):
#         ax = axes[i]
        
#         # --- 1. Kalman (灰色保持不变) ---
#         s1 = ax.scatter(kalman_samples[:, 0, t], kalman_samples[:, 1, t], 
#                    c='#7f7f7f', marker='o', alpha=0.3, s=15, linewidths=0, 
#                    label='Kalman Filter')
#         s2 = ax.scatter(kf_means[0, t], kf_means[1, t], 
#                    c='black', marker='*', s=120, edgecolors='white', linewidth=0.8,
#                    label='Kalman Mean', zorder=10)
        
#         # --- 2. Model (改为蓝色系) ---
#         # 散点颜色: #1f77b4 (经典的 Matplotlib 蓝 / Tableau Blue)
#         # s3 = ax.scatter(my_samples[:, 0, t], my_samples[:, 1, t], 
#         #            c='#1f77b4', marker='x', alpha=0.5, s=20, linewidths=1.0,
#         #            label='FBI-SSM (Ours)')
#         s3 = ax.scatter(my_samples[:, 0, t], my_samples[:, 1, t], 
#                    c= "blue",marker='x', alpha=0.5, s=20, linewidths=1.0,
#                    label='FBI-SSM (Ours)')
        
#         # 均值颜色: #003366 (深午夜蓝)，用于区分浅蓝色的散点，且显眼
#         s4 = ax.scatter(my_means[0, t], my_means[1, t], 
#                    c='#003366', marker='^', s=100, edgecolors='white', linewidth=0.8,
#                    label='Model Mean', zorder=11)
        
#         if i == 0:
#             handles.extend([s1, s3, s2, s4])
#             labels.extend(['Kalman Filter', 'FBI-SSM (Ours)', 'Kalman Mean', 'Model Mean'])

#         # 坐标轴美化
#         ax.set_xlim(xlims)
#         ax.set_ylim(ylims)
#         ax.set_title(f'Time Step $t={t+1}$', fontsize=14, fontweight='medium')
#         ax.set_xlabel('$x_1$', fontsize=12)
#         if i == 0:
#             ax.set_ylabel('$x_2$', fontsize=12)
        
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_alpha(0.6)
#         ax.spines['bottom'].set_alpha(0.6)
#         ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
#         ax.tick_params(axis='both', which='both', length=3, width=0.5, labelsize=10, color='gray')

#     # --- 图例 (保持柔和边框) ---
#     legend = fig.legend(handles, labels, 
#                loc='upper center',          
#                bbox_to_anchor=(0.5, 0.08),  
#                ncol=4,                      
#                columnspacing=2.5,           
#                handletextpad=0.5,           
#                fontsize=13,                 
#                markerscale=2.5,             
#                scatterpoints=1,             
               
#                frameon=True,                
#                fancybox=True,               
#                edgecolor='#C0C0C0',         
#                facecolor='white',           
#                framealpha=1.0,              
#                borderpad=0.8                
#                )
    
#     legend.get_frame().set_linewidth(0.5)

#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_2d_distribution_evolution(my_samples, kalman_samples, cols2plots=5, save_name=None):
    """
    绘制高颜值的学术风格 2D 分布演化图 (蓝色系)。
    改进点：使用学术蓝 (#1f77b4) 替换纯蓝，与 fill_between 风格统一。
    """
    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.linewidth'] = 0.8

    gen_size, p, T = my_samples.shape
    assert p == 2, "此函数仅适用于 p=2 的情况"
    
    my_samples = my_samples.detach().cpu().numpy()
    kalman_samples = kalman_samples.detach().cpu().numpy()
    
    my_means = np.mean(my_samples, axis=0)
    kf_means = np.mean(kalman_samples, axis=0)
    
    # 计算坐标范围
    all_data = np.concatenate([my_samples, kalman_samples], axis=0) 
    x_min, x_max = np.min(all_data[:, 0, :]), np.max(all_data[:, 0, :])
    y_min, y_max = np.min(all_data[:, 1, :]), np.max(all_data[:, 1, :])
    
    padding = (x_max - x_min) * 0.15
    # xlims = (x_min - padding, x_max + padding)
    # ylims = (y_min - padding, y_max + padding)
    xlims = (-20, 20)
    ylims = (-20, 20)
    
    if T <= cols2plots:
        selected_indices = np.arange(T)
        n_cols = T
    else:
        selected_indices = np.linspace(0, T-1, cols2plots, dtype=int)
        n_cols = cols2plots
    
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4), 
                             sharey=True, sharex=False, 
                             constrained_layout=True)
    
    if n_cols == 1: axes = np.array([axes])
    else: axes = axes.flatten()
    
    handles = []
    labels = []
    
    for i, t in enumerate(selected_indices):
        ax = axes[i]
        
        # --- 1. Kalman (灰色保持不变) ---
        s1 = ax.scatter(kalman_samples[:, 0, t], kalman_samples[:, 1, t], 
                   c='#7f7f7f', marker='o', alpha=0.3, s=15, linewidths=0, 
                   label='Kalman Filter')
        s2 = ax.scatter(kf_means[0, t], kf_means[1, t], 
                   c='black', marker='*', s=120, edgecolors='white', linewidth=0.8,
                   label='Kalman Mean', zorder=10)
        
        # --- 2. Model (改为学术蓝) ---
        # 为什么不用 'blue'? 因为纯蓝太刺眼。
        # 使用 '#1f77b4' (Tableau Blue)，这和 fill_between 默认的色系最接近，质感最好。
        s3 = ax.scatter(my_samples[:, 0, t], my_samples[:, 1, t], 
                #    c='#1f77b4',     # <--- 这里改为了最好看的学术蓝
                c = '#d62728',
                   marker='x', 
                   alpha=0.6,       # 稍微提高一点不透明度，让叉号更清晰
                   s=20, 
                   linewidths=1.0,
                   label='FBI-SSM (Ours)')
        
        # 均值颜色: 深蓝色 (#003366)，压住阵脚
        s4 = ax.scatter(my_means[0, t], my_means[1, t], 
                   c='#8b0000',     # 深红, marker='^', s=100, edgecolors='white', linewidth=0.8,
                   label='Model Mean', zorder=11)
        
        if i == 0:
            handles.extend([s1, s3, s2, s4])
            labels.extend(['Kalman Filter', 'FBI-SSM (Ours)', 'Kalman Mean', 'FBI-SSM Mean'])

        # 坐标轴美化
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(f'$t={t+1}$', fontsize=14, fontweight='medium')
        ax.set_xlabel('$x_1$', fontsize=12)
        if i == 0:
            ax.set_ylabel('$x_2$', fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.6)
        ax.spines['bottom'].set_alpha(0.6)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
        # ax.set_aspect('equal', 'box')
        ax.tick_params(axis='both', which='both', length=3, width=0.5, labelsize=10, color='gray')

    # --- 图例 (保持柔和边框) ---
    legend = fig.legend(handles, labels, 
               loc='upper center',          
               bbox_to_anchor=(0.5, 0.03),  
               ncol=4,                      
               columnspacing=2.5,           
               handletextpad=0.5,           
               fontsize=13,                 
               markerscale=2.5,             
               scatterpoints=1,             
               
               frameon=True,                
               fancybox=True,               
               edgecolor='#C0C0C0',         
               facecolor='white',           
               framealpha=1.0,              
               borderpad=0.8                
               )
    
    legend.get_frame().set_linewidth(0.5)

    # plt.show()
    if save_name is not None:
        plt.savefig("col"+str(cols2plots)+".pdf", format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.05)
        
        
        
#         # ==========================================
def visualize_100_samples(X_complete_list, p, T):
    """
    输入: X_complete_list (list of Tensors)
    输出: 绘制每个 Feature 的中位数和 95% 置信区间
    """
    # 1. 堆叠数据: (Num_Samples, p, T)
    # 例如 (100, 4, 50)
    data_stack = torch.stack(X_complete_list).detach().cpu().numpy()
    
    # 2. 计算统计量 (沿着 Sample 维度 0)
    # 95% CI -> 2.5% 到 97.5%
    median = np.median(data_stack, axis=0)      # (p, T)
    lower = np.percentile(data_stack, 5, axis=0) # (p, T)
    upper = np.percentile(data_stack, 95, axis=0) # (p, T)
    
    # 3. 绘图
    time_steps = np.arange(T)
    
    fig, axes = plt.subplots(p, 1, figsize=(10, 3*p), sharex=True)
    if p == 1: axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 不同颜色区分不同Feature
    
    print(f"Plotting stats for {len(X_complete_list)} samples...")
    
    for i in range(p):
        ax = axes[i]
        c = colors[i % len(colors)]
        
        # A. 绘制置信区间 (95%)
        ax.fill_between(time_steps, lower[i], upper[i], color=c, alpha=0.2, label='95% CI')
        
        # B. 绘制中位数
        ax.plot(time_steps, median[i], color=c, linewidth=2, label='Median Trend')
        
        # C. (可选) 随机画 3 条具体的样本轨迹，增加直观感
        for sample_idx in np.random.choice(len(X_complete_list), 10, replace=False):
            ax.plot(time_steps, data_stack[sample_idx, i, :], color=c, alpha=0.3, linewidth=0.8, linestyle='--')

        ax.set_title(f'Feature {i+1} Dynamics', fontweight='bold', loc='left')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.xlabel('Time Step')
    plt.suptitle(f'Dynamics of {p} Features over 100 Samples', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()