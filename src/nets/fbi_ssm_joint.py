"""
FBI-SSM for Joint Training (Algorithm 3)

🔥 This module is identical to fbi_ssm_physionet2012.py
   The difference is in the training procedure (train_joint.py)
   
🔥 Joint Training Algorithm:
   - No separate phases for (f,g) and (h,u,q)
   - Single integrated loop with joint loss:
     Total_Loss = L_forecast(f, g, h) + λ * L_selfmask(u, q, h)
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append("..")

# Import all components from net_physionet2012
from nets.net_physionet2012 import SSM_Filter, SSM_Decoder_U, SSM_Smoother, InitialNoiseProcessor, InitialNoiseProcessorLearnable
from nets.net_physionet2012 import SSM_Out_Fx, SSM_Out_Fx_MLP, SSM_Out_Fx_Adaptive, SSM_Out_Fo, SSM_Out_Fo_MLP, SSM_Out_Fo_Constant, SSM_Out_Fo_Oracle, SSM_Out_Fo_OracleFixed, SSM_Out_Fo_Blend, SSM_Transition_G
from nets.net_physionet2012 import SSM_Filter_V2, SSM_Smoother_V2
from nets.net_physionet2012 import SSM_Smoother_V3
from nets.net_physionet2012 import SSM_Smoother_V4
from nets.net_physionet2012 import SSM_Smoother_V5
from nets.net_physionet2012 import SSM_Smoother_V6
from nets.net_physionet2012 import SSM_Decoder_V2


class FBI_SSM_Joint(nn.Module):
    """
    FBI-SSM Model for Joint Training (Algorithm 3).
    
    Same architecture as FBI_SSM in fbi_ssm_physionet2012.py,
    but designed for joint training where all components (f,g,h,u,q)
    are updated simultaneously.
    """
    def __init__(self, p, T=1, d=1, device=0, learn_initial_dist=True, mu_z1=None, sigma_z1=None,
                 hidden_scale=1.0, fx_type='mlp', fo_type='mlp', is_complete_data=False,
                 modality_dropout=0.0, force_alpha=None, encoder_version='v2',
                 use_alibi=False, smoother_version='v1', hard_replace_u=True,
                 decoder_type='u', no_mask_input=False, fo_kwargs=None,
                 use_mask_decoder=False):
        """
        FBI-SSM Model for Joint Training.
        
        Args:
            p: int, feature dimension (35 for PhysioNet 2012)
            T: int, time dimension (48 for PhysioNet 2012)
            d: int, latent dimension (keep_dim)
            device: device to use
            learn_initial_dist: bool, if True use learnable MLP for z1 distribution,
                               if False use theoretical distribution (default)
            mu_z1: (d,) tensor, theoretical mean of z1 (from calculate_burn_in_distribution)
            sigma_z1: (d, d) tensor, theoretical covariance of z1
            hidden_scale: float, scale factor for hidden dimensions (default=1.0)
            fx_type: str, observation model type:
                     'linear' for SSM_Out_Fx (with linear shortcut)
                     'mlp' for SSM_Out_Fx_MLP (pure MLP)
                     'adaptive' for SSM_Out_Fx_Adaptive
            fo_type: str, mask model type:
                     'default' for SSM_Out_Fo 
                     'mlp' for SSM_Out_Fo_MLP
            is_complete_data: bool, if True, out_fo will always output 1.0
            modality_dropout: float, dropout rate for spatial modality during training
            force_alpha: float or None, force alpha value in decoder_u
            encoder_version: str, 'v1' or 'v2'
            use_alibi: bool, whether to use ALiBi positional encoding
            smoother_version: str, 'v1', 'v2', 'v3', 'v4', 'v5', or 'v6'
            hard_replace_u: bool, hard replacement at observed positions
            decoder_type: str, 'u' for SSM_Decoder_U, 'v2' for SSM_Decoder_V2 (Attention+FFN)
        """
        super().__init__()
        
        self.keep_dim = d
        self.seq_len = T
        self.feature_dim = p
        self.time_dim = T
        self.device = device
        self.learn_initial_dist = learn_initial_dist
        self.hidden_scale = hidden_scale
        self.fx_type = fx_type
        self.fo_type = fo_type
        self.is_complete_data = is_complete_data
        self.modality_dropout = modality_dropout
        self.force_alpha = force_alpha
        self.encoder_version = encoder_version.lower()
        self.use_alibi = use_alibi
        self.smoother_version = smoother_version.lower()
        self.hard_replace_u = hard_replace_u
        self.decoder_type = decoder_type.lower()
        self.no_mask_input = no_mask_input
        self.use_mask_decoder = use_mask_decoder
        
        self.noise_dim_latent = self.keep_dim
        self.noise_dim = self.feature_dim
        self.noise_scale = 1.0
        
        # Apply hidden_scale to control network complexity
        n_head = max(2, int(4 * hidden_scale))
        raw_d_model = 128
        self.d_model = ((raw_d_model + n_head - 1) // n_head) * n_head
        d_inner = 512
        d_k = max(8, int(32 * hidden_scale))
        d_v = max(16, int(32 * hidden_scale))
        dropout = 0
        self.dropout = nn.Dropout(p=dropout)
        n_groups = 3
        self.n_group_inner_layers = 2
        
        kwargs = {}
        kwargs["device"] = device
        kwargs["diagonal_attention_mask"] = True 
        
        # --- COMPONENT: Initial State (noise processor) ---
        if learn_initial_dist:
            self.noise_processor = InitialNoiseProcessorLearnable(self.keep_dim)
        else:
            self.noise_processor = InitialNoiseProcessor(self.keep_dim, mu=mu_z1, sigma=sigma_z1)
        
        # --- COMPONENT: Encoder h ---
        if self.encoder_version == 'v2':
            self.encoder_h = SSM_Filter_V2(
                    self.time_dim,
                    self.feature_dim,
                    self.keep_dim,
                    self.d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    n_groups,
                    self.n_group_inner_layers,
                    **kwargs
                )
            print(f"✅ [Joint] Using SSM_Filter_V2 (双向注意力: time + feature)")
        else:
            self.encoder_h = SSM_Filter(
                    self.time_dim,
                    self.feature_dim,
                    self.keep_dim,
                    self.d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    n_groups,
                    self.n_group_inner_layers,
                    **kwargs
                )
            print(f"✅ [Joint] Using SSM_Filter (causal mask)")
        
        # --- COMPONENT: Output f (observation) ---
        if fx_type == 'mlp':
            self.out_fx = SSM_Out_Fx_MLP(self.feature_dim, self.keep_dim, self.d_model)
        elif fx_type == 'adaptive':
            self.out_fx = SSM_Out_Fx_Adaptive(self.feature_dim, self.keep_dim, self.d_model)
            print(f"✅ [Joint] Using Adaptive Fx")
        else:
            self.out_fx = SSM_Out_Fx(self.feature_dim, self.keep_dim, self.d_model)
        
        # --- COMPONENT: Output f_o (mask prediction) ---
        if fo_type == 'mlp':
            self.out_fo = SSM_Out_Fo_MLP(self.feature_dim, self.keep_dim, self.d_model)
            print(f"✅ [Joint] Using MLP Fo")
        elif fo_type == 'mlp_zero_z':
            self.out_fo = SSM_Out_Fo_MLP(self.feature_dim, self.keep_dim, self.d_model, zero_z=True)
            print(f"✅ [Joint] Using MLP Fo with zero_z (same params, no Z-dependency)")
        elif fo_type == 'all_one':
            self.out_fo = SSM_Out_Fo_MLP(self.feature_dim, self.keep_dim, self.d_model, all_one=True)
            print(f"✅ [Joint] Using MLP Fo with all_one (same params, output always 1)")
        elif fo_type == 'oracle':
            self.out_fo = SSM_Out_Fo_Oracle(self.feature_dim, self.keep_dim, self.d_model)
            print(f"✅ [Joint] Using Oracle Fo (linear W@Z+b, matches DGP)")
        elif fo_type == 'constant':
            self.out_fo = SSM_Out_Fo_Constant(self.feature_dim, self.keep_dim, self.d_model)
            print(f"✅ [Joint] Using Constant Fo (no informative missingness)")
        elif fo_type == 'oracle_gt':
            fo_kw = fo_kwargs or {}
            self.out_fo = SSM_Out_Fo_OracleFixed(
                self.feature_dim, self.keep_dim, self.d_model,
                gt_weight=fo_kw.get('gt_weight'), gt_bias=fo_kw.get('gt_bias'),
                freeze=fo_kw.get('freeze', False))
            frozen_str = "frozen" if fo_kw.get('freeze', False) else "trainable"
            print(f"✅ [Joint] Using OracleFixed Fo (GT-initialized, {frozen_str})")
        elif fo_type == 'blend':
            fo_kw = fo_kwargs or {}
            self.out_fo = SSM_Out_Fo_Blend(
                self.feature_dim, self.keep_dim, self.d_model,
                alpha=fo_kw.get('alpha', 0.0),
                beta=fo_kw.get('beta', 2.5),
                c_intercept=fo_kw.get('c_intercept', 0.0),
                severity_dim=fo_kw.get('severity_dim', 0),
                b_vec=fo_kw.get('b_vec', None),
            )
            _a = fo_kw.get('alpha', 0.0)
            print(f"✅ [Joint] Using Blend Fo (alpha={_a:.2f}, severity_dim={fo_kw.get('severity_dim', 0)})")
        else:
            self.out_fo = SSM_Out_Fo(self.feature_dim, self.keep_dim, self.d_model)
        
        # --- COMPONENT: Transition g ---
        self.transition_g = SSM_Transition_G(self.keep_dim, self.d_model)
        
        # --- COMPONENT: Decoder u ---
        if self.decoder_type == 'v2':
            self.decoder_u = SSM_Decoder_V2(
                feature_dim=self.feature_dim, 
                z_dim=self.keep_dim, 
                d_model=self.d_model,
                n_heads=n_head,
                dropout=dropout,
            )
            print(f"✅ [Joint] Using SSM_Decoder_V2 (Attention+FFN)")
        else:
            self.decoder_u = SSM_Decoder_U(
                self.feature_dim, self.keep_dim, self.d_model,
                modality_dropout=self.modality_dropout,
                force_alpha=self.force_alpha,
                hard_replace_u=self.hard_replace_u
            )
            # Print config
            if self.force_alpha is not None:
                print(f"✅ [Joint] decoder_u force_alpha={self.force_alpha} (ablation mode)")
            else:
                print(f"✅ [Joint] decoder_u modality_dropout={self.modality_dropout}")
            print(f"✅ [Joint] decoder_u hard_replace_u={self.hard_replace_u}")
        
        if self.no_mask_input:
            print(f"🚫 [Joint] no_mask_input=True → encoder sees mask=1 (no mask information)")
        
        # --- COMPONENT: Smoother q ---
        if self.smoother_version == 'v6':
            self.backward_transition_q = SSM_Smoother_V6(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout)
            print(f"✅ [Joint] Using SSM_Smoother_V6 (Non-Markovian: Q=z_f, K/V=Ž_future)")
        elif self.smoother_version == 'v5':
            self.backward_transition_q = SSM_Smoother_V5(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout)
            print(f"✅ [Joint] Using SSM_Smoother_V5")
        elif self.smoother_version == 'v4':
            self.backward_transition_q = SSM_Smoother_V4(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout)
            print(f"✅ [Joint] Using SSM_Smoother_V4 (Cross Attention: Q=Z_next, K/V=Z_f)")
        elif self.smoother_version == 'v3':
            self.backward_transition_q = SSM_Smoother_V3(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                k_order=1,
                use_alibi=use_alibi)
            print(f"✅ [Joint] Using SSM_Smoother_V3 (符合理论: 使用原始观测 X^o, ALiBi={use_alibi})")
        elif self.smoother_version == 'v2':
            self.backward_transition_q = SSM_Smoother_V2(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout)
            print(f"✅ [Joint] Using SSM_Smoother_V2 (近似: 使用 encoder 输出 H)")
        else:
            self.backward_transition_q = SSM_Smoother(
                feature_dim=p, 
                keep_dim=d, 
                time_dim=T, 
                d_model=self.d_model, 
                n_head=4,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                transition_g=self.transition_g)
            print(f"✅ [Joint] Using SSM_Smoother_V1")
        
        # --- COMPONENT: Mask Decoder u_o (optional) ---
        # Reuses out_fo (Z_s → probs) instead of a separate linear layer;
        # trained with BCE vs original mask.
        self.mask_decoder_u = None   # kept for API compatibility (always None)
        if self.use_mask_decoder:
            print(f"✅ [Joint] mask_decoder_u enabled (reuses out_fo, BCE loss)")
    
    def set_initial_distribution(self, mu, sigma):
        """Set z1 theoretical distribution parameters"""
        if self.learn_initial_dist:
            print("Warning: learn_initial_dist=True, set_initial_distribution has no effect")
            return
        self.noise_processor.set_distribution(mu, sigma)
        print(f"Initial distribution updated: mu shape={mu.shape}, sigma shape={sigma.shape}")

    def run_forward_pass(self, x, mask, train_flag=True, time_stamps=None):
        """
        Forward pass through encoder h and transition g.
        
        Args:
            x: (B, T, p) observed data
            mask: (B, T, p) observation mask
            train_flag: whether in training mode
            time_stamps: (B, T) or (T,) or None
                
        Returns:
            forecasting_X1, forecasting_X2: (B, T, p) two independent samples
            forecasting_O1, forecasting_O2: (B, T, p) predicted masks
            Z_full_encoded: (B, T, d) encoder output
        """
        # If no_mask_input=True, replace mask with ones so encoder gets no mask info
        mask_for_encoder = torch.ones_like(mask) if self.no_mask_input else mask
        if hasattr(self.encoder_h, 'forward') and 'time_stamps' in self.encoder_h.forward.__code__.co_varnames:
            Z_full_encoded = self.encoder_h(x, mask_for_encoder, time_stamps=time_stamps)
        else:
            Z_full_encoded = self.encoder_h(x, mask_for_encoder)
        B, T_enc, keep_dim = Z_full_encoded.shape
        
        # transition_g is deterministic: both samples share the same output.
        # Independence between the two forecast samples comes solely from the
        # stochastic z1 draws below, so one g-call suffices.
        z_input = Z_full_encoded[:, :-1, :] if train_flag else Z_full_encoded
        pred_Z_shared = self.transition_g(z_input)
        
        # Add z1 to complete T steps (the two samples differ only here)
        learned_z1_prime1 = self.noise_processor(B, Z_full_encoded.device)
        learned_z1_prime2 = self.noise_processor(B, Z_full_encoded.device)
        pred_Z1 = torch.cat([learned_z1_prime1, pred_Z_shared], dim=1)
        pred_Z2 = torch.cat([learned_z1_prime2, pred_Z_shared], dim=1)
        
        forecasting_X1 = self.out_fx(pred_Z1)
        forecasting_X2 = self.out_fx(pred_Z2)
        
        if self.is_complete_data:
            forecasting_O1 = torch.ones_like(forecasting_X1)
            forecasting_O2 = torch.ones_like(forecasting_X2)
        else:
            forecasting_O1 = self.out_fo(pred_Z1)
            forecasting_O2 = self.out_fo(pred_Z2)
        
        return forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, Z_full_encoded

    def forecasting(self, X, Omega_b, gen_sample_size=1, train_flag=True, extra_steps=1, time_stamps=None):
        """
        Forecasting mode: forward pass only (no backward smoothing).
        
        Args:
            X: (batch_size, p, T) observed data
            Omega_b: (batch_size, p, T) observation mask
            gen_sample_size: number of samples to generate
            train_flag: whether in training mode
            extra_steps: extra prediction steps
            time_stamps: (batch_size, T) or (T,) or None
        """
        batch_size, p, T = X.shape
        if train_flag:
            T_total = T
        else:
            T_total = T + extra_steps
        
        if gen_sample_size > 1:
            X = X.unsqueeze(1).expand(-1, gen_sample_size, -1, -1).reshape(-1, p, T)
            Omega_b = Omega_b.unsqueeze(1).expand(-1, gen_sample_size, -1, -1).reshape(-1, p, T)
            if time_stamps is not None:
                if time_stamps.dim() == 1:
                    pass
                else:
                    time_stamps = time_stamps.unsqueeze(1).expand(-1, gen_sample_size, -1).reshape(-1, T)
        
        X_permuted = X.permute(0, 2, 1)
        Omega_b_permuted = Omega_b.permute(0, 2, 1)
        
        forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, Z_f = self.run_forward_pass(
            X_permuted, Omega_b_permuted, train_flag, time_stamps=time_stamps
        )
        
        forecasting_X1 = forecasting_X1.permute(0, 2, 1)
        forecasting_X2 = forecasting_X2.permute(0, 2, 1)
        forecasting_X1 = forecasting_X1.view(batch_size, gen_sample_size, p, T_total)
        forecasting_X2 = forecasting_X2.view(batch_size, gen_sample_size, p, T_total)
        
        forecasting_O1 = forecasting_O1.permute(0, 2, 1)
        forecasting_O2 = forecasting_O2.permute(0, 2, 1)
        forecasting_O1 = forecasting_O1.view(batch_size, gen_sample_size, p, T_total)
        forecasting_O2 = forecasting_O2.view(batch_size, gen_sample_size, p, T_total)
        
        return forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, Z_f
        
    def run_backward_pass(self, Z_f, X_obs=None, mask_obs=None, time_stamps=None):
        """
        Backward pass (smoothing): inference mode, using own Z_s as z_next.
        
        Args:
            Z_f: [B, T, keep_dim] - encoder output
            X_obs: [B, T, p] - original observation (V3 requires)
            mask_obs: [B, T, p] - observation mask (V3 requires)
            time_stamps: [B, T] or [T,] or None - timestamps
            
        Returns:
            Z_s: [B, T, keep_dim] - smoothed latent
        """
        batch_size_gen, T, d = Z_f.shape
        
        z_list = [None] * T
        z_list[T-1] = Z_f[:, T-1, :]

        # V6: Non-Markovian smoother (Q=z_f, K/V=Ž_future)
        if hasattr(self.backward_transition_q, 'precompute_forward_features'):
            self.backward_transition_q.precompute_forward_features(Z_f, time_stamps)
            self.backward_transition_q.initialize_boundary()  # Ž_{T-1} = z_{T-1}^f
            
            for t in range(T - 2, -1, -1):
                z_curr = self.backward_transition_q.forward_step(t)
                z_list[t] = z_curr
            
            self.backward_transition_q.clear_cache()
        # V3: uses original observations
        elif hasattr(self.backward_transition_q, 'precompute_observation_features'):
            assert X_obs is not None and mask_obs is not None, \
                "V3 Smoother requires X_obs and mask_obs for backward pass!"
            self.backward_transition_q.precompute_observation_features(X_obs, mask_obs, time_stamps)
            
            for t in range(T - 2, -1, -1):
                z_next_t = z_list[t+1]
                z_future = z_next_t.unsqueeze(1)
                z_curr = self.backward_transition_q.forward_with_cache(t, z_future)
                z_list[t] = z_curr
            
            self.backward_transition_q.clear_cache()
        # V2/V1: uses encoder output H
        elif hasattr(self.backward_transition_q, 'precompute_history_features'):
            self.backward_transition_q.precompute_history_features(Z_f, time_stamps)
            
            is_v1_simple = hasattr(self.backward_transition_q, 'gain_net') or \
                           hasattr(self.backward_transition_q, 'residual_mlp')
            
            for t in range(T - 2, -1, -1):
                z_s_next = z_list[t+1]
                
                if is_v1_simple:
                    z_f_t = Z_f[:, t, :]
                    z_curr = self.backward_transition_q(z_f_t, z_s_next)
                else:
                    z_curr = self.backward_transition_q.forward_with_cache(t, z_s_next)
                
                z_list[t] = z_curr
            
            self.backward_transition_q.clear_cache()
        else:
            # Fallback
            z_next = Z_f[:, -1, :]
            for t in range(T - 2, -1, -1):
                z_f_t = Z_f[:, t, :]
                try:
                    z_next = self.backward_transition_q(z_f_t, z_next)
                except TypeError:
                    H = Z_f[:, :t + 1, :]
                    z_next = self.backward_transition_q(H, z_next)
                z_list[t] = z_next
        
        Z_s = torch.stack(z_list, dim=1)
            
        return Z_s
    
    def imputation(self, X, Omega_b, Z_f, gen_sample_size=1, keep_dim=None, time_stamps=None):
        """
        Imputation mode: full forward + backward pass.
        
        Args:
            X: [B, p, T] - original observation
            Omega_b: [B, p, T] - observation mask
            Z_f: [B*gen_sample_size, T, keep_dim] - encoder output
            gen_sample_size: number of samples
            time_stamps: [B, T] or [T,] or None
        """
        batch_size, _, _ = X.shape
        if gen_sample_size > 1:
            X = X.unsqueeze(1).expand(-1, gen_sample_size, -1, -1).reshape(-1, self.feature_dim, self.seq_len)
            Omega_b = Omega_b.unsqueeze(1).expand(-1, gen_sample_size, -1, -1).reshape(-1, self.feature_dim, self.seq_len)
        
        X_permuted = X.permute(0, 2, 1)
        Omega_b_permuted = Omega_b.permute(0, 2, 1)
        
        Z_s = self.run_backward_pass(Z_f, X_obs=X_permuted, mask_obs=Omega_b_permuted, time_stamps=time_stamps)
        
        # decoder_u is deterministic (dropout=0): both samples are identical,
        # so compute once and reuse.
        smooth_X1 = self.decoder_u(Z_s, X_permuted, Omega_b_permuted)
        smooth_X2 = smooth_X1
        
        smooth_X1 = smooth_X1.permute(0, 2, 1)
        smooth_X2 = smooth_X2.permute(0, 2, 1)
        
        Omega_b = Omega_b.float()
        smooth_X1 = Omega_b * X + (1 - Omega_b) * smooth_X1
        smooth_X2 = Omega_b * X + (1 - Omega_b) * smooth_X2
        smooth_X1 = smooth_X1.view(batch_size, gen_sample_size, self.feature_dim, self.seq_len)
        smooth_X2 = smooth_X2.view(batch_size, gen_sample_size, self.feature_dim, self.seq_len)
        
        return smooth_X1, smooth_X2, Omega_b
    
    def forward(self, X, Omega_b, gen_sample_size=1, **kwargs):
        """Complete inference: Forecasting + Imputation"""
        forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, Z_f = self.forecasting(X, Omega_b, gen_sample_size)
        smooth_X1, smooth_X2, Omega_b = self.imputation(X, Omega_b, Z_f, gen_sample_size)
        
        return forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, smooth_X1, smooth_X2, Omega_b, Z_f

    def generate_synthetic_data_from_fg(self, synthetic_size=1, z_clamp=10.0):
        """Generate synthetic data using f, g (for evaluation/comparison).

        WARNING: transition_g is unconstrained (no spectral-radius constraint).
        The joint training loss is computed on encoder outputs — NOT on free
        autoregressive rollouts — so the model receives no gradient signal that
        would enforce rollout stability.  If transition_g Jacobian eigenvalues
        exceed 1, the Z sequence will diverge over the T time steps.

        The `z_clamp` parameter clips Z at each step to prevent blow-up.  If you
        see Z_max growing toward the clamp value the transition is near-unstable.
        """
        h_0 = self.noise_processor(synthetic_size, self.device)
        syn_seq_len = self.seq_len

        # Pre-allocate lists to avoid O(T²) torch.cat inside the loop
        Z_list = [h_0]
        X_list = [self.out_fx(h_0)]
        O_list = [torch.bernoulli(self.out_fo(h_0))]

        for t in range(1, syn_seq_len):
            Z_t = self.transition_g(Z_list[-1])
            if Z_t.dim() == 2:
                Z_t = Z_t.unsqueeze(1)

            # Clip to prevent diverging autoregressive rollouts
            Z_t = torch.clamp(Z_t, min=-z_clamp, max=z_clamp)

            Z_list.append(Z_t)
            X_list.append(self.out_fx(Z_t))
            O_list.append(torch.bernoulli(self.out_fo(Z_t)))

        Z_syn = torch.cat(Z_list, dim=1)
        X_syn = torch.cat(X_list, dim=1)
        O_syn = torch.cat(O_list, dim=1)

        return X_syn, O_syn, Z_syn


# Backward compatibility alias
FBI_SSM = FBI_SSM_Joint
