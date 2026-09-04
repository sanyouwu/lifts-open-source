"""
FBI (Factor-Based Imputation) Wrapper

This wrapper provides a unified interface for FBI imputation, matching the
interface of CSDIImputer and other baseline methods.

FBI uses a generative factor model with state-space structure:
- Encoder (h): Maps observations to latent factors Z
- Transition (g): Models temporal dynamics in latent space
- Output (f): Maps latent factors to observations
- Smoother (q): Backward pass for smoothing
- Decoder (u): Final imputation conditioned on observed values
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Add FBI directory to path
FBI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FBI_DIR)


def get_fbi_config(p=20, T=48, d=None, obs_rate=0.5):
    """
    Get default FBI configuration.
    
    Args:
        p: int, feature dimension
        T: int, time dimension
        d: int, latent dimension (keep_dim). Default None = use p (full rank)
        obs_rate: float, expected observation rate (for config naming)
    
    Returns:
        dict with 'model', 'train', and 'data' configurations
    """
    # Default latent dimension to p (full rank) if not specified
    d = d if d is not None else p
    
    config = {
        'model': {
            'd': d,                      # latent dimension (default: p)
            'learn_initial_dist': True,  # learn initial z1 distribution
            'hidden_scale': 1.0,         # network hidden dimension scale
            'fx_type': 'mlp',            # 'linear', 'mlp', or 'adaptive'
            'fo_type': 'mlp',            # 'default' or 'mlp'
            'is_complete_data': False,   # set True if obs_rate=1.0
            'encoder_type': 'v1',        # 'v1' or 'v2' for encoder version
            'smoother_type': 'v1',       # 'v1', 'v2', 'v3', 'v4', 'v5', or 'v6'
            'decoder_type': 'v2',        # 'u' for SSM_Decoder_U, 'v2' for SSM_Decoder_V2
            'use_mask_decoder': False,   # enable u_o mask decoder on smoothed Z
        },
        'train': {
            'lr_fg': 1e-3,               # learning rate for f, g
            'lr_huq': 1e-3,              # learning rate for h, u, q
            'epochs_fg': 1,              # epochs per alternating loop for f, g
            'epochs_huq': 1,             # epochs per alternating loop for h, u, q
            'total_alternating_loops': 100,  # Phase 1: total training loops for f,g,h,q
            'epochs_u': 100,             # Phase 2: epochs for training decoder u only
            'batch_size_fg': 128,        # batch size for f, g training (real data)
            'batch_size_huq': 128,       # batch size for h, q training (synthetic data)
            'batch_size_u': 128,         # batch size for decoder u training (Phase 2)
            'gen_sample_size': 1,        # samples for energy loss during training
            'alpha': 1.0,                # weight for energy score
            'syn_size': None,            # synthetic data size for huq training (None = use train size)
            'drop_rate': 0.2,            # dropout rate for self-mask training
            'train_u': 'self_mask',      # 'fg', '0', or 'self_mask'
            'lambda_fo': 1.0,            # weight for mask prediction loss
            'lambda_fx': 1.0,            # weight for data prediction loss (0 = mask-only training)
            'fo_loss_type': 'l2',        # 'l2' (original) or 'bce' (binary cross-entropy for mask)
            'lambda_selfmask': 5.0,      # weight for self-mask imputation loss
            'lambda_d_o': 0.0,           # weight for u_o mask decoder BCE loss (0 = disabled)
            'mask_dec_pos_weight': None, # pos_weight for BCE mask loss (e.g. 4.0 for 20% obs rate)
            'lambda_enc': 0.0,           # weight for synthetic encoder consistency loss (0 = disabled)
            'enc_warmup': 50,            # epochs before enabling encoder consistency
            'enc_syn_batch': None,       # synthetic trajectories per L_enc step (None = min(4000, N))
            'enc_syn_mini_batch': None,  # chunk size for L_enc synthesis (None = min(batch_size, enc_syn_batch))
            'enc_noise_scale': 0.0,      # optional noise on encoder samples for L_enc
            'enc_ema_decay': 0.999,      # EMA momentum for synthetic-data generator
        },
        'data': {
            'p': p,
            'T': T,
            'd': d,  # latent dimension (default: p)
            'obs_rate': obs_rate,
        }
    }
    return config


class _EncoderConsistencyEMA:
    """EMA copy of the generator modules used for stop-gradient L_enc targets."""

    _GENERATOR_ATTRS = ('transition_g', 'out_fx', 'out_fo', 'noise_processor')

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for attr in self._GENERATOR_ATTRS:
            shadow_module = getattr(self.shadow, attr, None)
            live_module = getattr(model, attr, None)
            if shadow_module is None or live_module is None:
                continue
            for shadow_param, live_param in zip(shadow_module.parameters(), live_module.parameters()):
                shadow_param.data.mul_(self.decay).add_(live_param.data, alpha=1.0 - self.decay)
            for shadow_buffer, live_buffer in zip(shadow_module.buffers(), live_module.buffers()):
                shadow_buffer.data.copy_(live_buffer.data)


def _compute_encoder_consistency_loss(model, ema, enc_syn_total, enc_syn_mini_batch,
                                      enc_noise_scale, device):
    """Energy-score L_enc that calibrates encoder_h on EMA-generated synthetic data."""
    def _l2_sum(left, right):
        return torch.sqrt(((left - right) ** 2).sum(dim=(-2, -1)) + 1e-8).sum()

    acc_target_1 = torch.tensor(0.0, device=device)
    acc_target_2 = torch.tensor(0.0, device=device)
    acc_pair = torch.tensor(0.0, device=device)
    n_done = 0

    while n_done < enc_syn_total:
        chunk_size = min(enc_syn_mini_batch, enc_syn_total - n_done)

        with torch.no_grad():
            X_syn, O_syn, Z_syn = ema.shadow.generate_synthetic_data_from_fg(
                synthetic_size=chunk_size
            )
            X_syn_obs = (X_syn * O_syn).to(device)
            O_syn_dev = O_syn.to(device)
            Z_true = Z_syn.to(device)

        O_syn_enc = torch.ones_like(O_syn_dev) if model.no_mask_input else O_syn_dev
        Z_enc_1 = model.encoder_h(X_syn_obs, O_syn_enc)
        Z_enc_2 = model.encoder_h(X_syn_obs, O_syn_enc)

        if enc_noise_scale > 0.0:
            Z_enc_1 = Z_enc_1 + enc_noise_scale * torch.randn_like(Z_enc_1)
            Z_enc_2 = Z_enc_2 + enc_noise_scale * torch.randn_like(Z_enc_2)

        acc_target_1 = acc_target_1 + _l2_sum(Z_enc_1, Z_true)
        acc_target_2 = acc_target_2 + _l2_sum(Z_enc_2, Z_true)
        acc_pair = acc_pair + _l2_sum(Z_enc_1, Z_enc_2)
        n_done += chunk_size

    loss_enc = (acc_target_1 + acc_target_2 - acc_pair) / n_done
    return loss_enc


class FBIImputer:
    """
    FBI Imputer for time-series data.
    
    Args:
        config: dict, model configuration (from get_fbi_config)
        device: torch device
        n_samples: int, number of imputation samples (default: 100)
        seed: int, random seed
        
    Note:
        Training parameters (total_loops, batch_size, etc.) are read from config['train'].
        Use get_fbi_config() to get a default config, then modify as needed.
    """
    
    def __init__(self, config=None, device=None, n_samples=200, seed=None,
                 # Legacy params (deprecated, use config instead)
                 total_loops=None, batch_size=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_samples = n_samples
        self.seed = seed
        
        # Handle legacy params (override config if provided)
        if total_loops is not None and self.config:
            self.config['train']['total_alternating_loops'] = total_loops
        if batch_size is not None and self.config:
            self.config['train']['batch_size'] = batch_size
        
        # Get params from config
        train_cfg = self.config.get('train', {}) if self.config else {}
        self.total_loops = train_cfg.get('total_alternating_loops', 120)
        self.batch_size = train_cfg.get('batch_size', 128)
        
        self.model = None
        self.scaler = None
        self.mean_scaler = None
        self.trained = False

    def load_from_checkpoint(self, ckpt_path, p, T, mean_scaler=None, scaler=None):
        """
        Load a pre-trained model from a checkpoint file.

        The checkpoint may be:
          - A plain state_dict (saved with torch.save(model.state_dict(), path))
          - A dict with keys 'model_state_dict', and optionally 'mean_scaler'/'scaler'

        Args:
            ckpt_path: str or Path, path to the .pth checkpoint file
            p: int, feature dimension (must match saved model)
            T: int, time dimension (must match saved model)
            mean_scaler: np.array shape (p,) or (1,p,1). If None, uses zeros (identity).
            scaler: np.array shape (p,) or (1,p,1). If None, uses ones (identity).
        """
        self.shape = (p, T)

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Handle both plain state_dict and wrapped checkpoint dicts
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            if mean_scaler is None and 'mean_scaler' in ckpt:
                mean_scaler = ckpt['mean_scaler']
            if scaler is None and 'scaler' in ckpt:
                scaler = ckpt['scaler']
            # Apply stored model_cfg so _create_model builds the correct architecture.
            # The checkpoint uses 'smoother_version'/'encoder_version'; the config
            # uses 'smoother_type'/'encoder_type' — map accordingly.
            if 'model_cfg' in ckpt:
                if self.config is None:
                    self.config = get_fbi_config(p=p, T=T)
                mc = self.config.setdefault('model', {})
                key_map = {'smoother_version': 'smoother_type',
                           'encoder_version': 'encoder_type'}
                for ck, cv in ckpt['model_cfg'].items():
                    mc[key_map.get(ck, ck)] = cv
                print(f"  [load_from_checkpoint] Applied model_cfg from checkpoint "
                      f"(smoother={mc.get('smoother_type')}, "
                      f"encoder={mc.get('encoder_type')}, "
                      f"decoder={mc.get('decoder_type')})")
        else:
            state_dict = ckpt

        # Build model AFTER applying checkpoint's model_cfg
        self.model = self._create_model(p, T)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"  [load_from_checkpoint] Missing keys: {missing}")
        if unexpected:
            print(f"  [load_from_checkpoint] Unexpected keys: {unexpected}")

        # Set normalization scalers
        if mean_scaler is not None:
            self.mean_scaler = np.array(mean_scaler, dtype=np.float32).reshape(1, p, 1)
        else:
            self.mean_scaler = np.zeros((1, p, 1), dtype=np.float32)

        if scaler is not None:
            self.scaler = np.array(scaler, dtype=np.float32).reshape(1, p, 1)
            self.scaler[self.scaler == 0] = 1.0
        else:
            self.scaler = np.ones((1, p, 1), dtype=np.float32)

        self.trained = True
        print(f"  [load_from_checkpoint] Loaded model from {ckpt_path} (p={p}, T={T})")

    def _create_model(self, p, T):
        """Create FBI model (using FBI_SSM_Joint for Joint Training)"""
        from nets.fbi_ssm_joint import FBI_SSM_Joint
        
        # Get config
        if self.config is None:
            self.config = get_fbi_config(p=p, T=T)
        
        model_cfg = self.config.get('model', {})
        d = model_cfg.get('d', p)  # default: full rank (p)
        learn_initial_dist = model_cfg.get('learn_initial_dist', True)
        hidden_scale = model_cfg.get('hidden_scale', 1.0)
        fx_type = model_cfg.get('fx_type', 'mlp')  # default: mlp
        fo_type = model_cfg.get('fo_type', 'mlp')  # default: mlp
        is_complete_data = model_cfg.get('is_complete_data', False)
        encoder_type = model_cfg.get('encoder_type', 'v1')  # default: v1
        smoother_type = model_cfg.get('smoother_type', 'v1')  # default: v1
        decoder_type = model_cfg.get('decoder_type', 'v2')  # default: v2
        hard_replace_u = model_cfg.get('hard_replace_u', True)  # 🔥 添加 hard_replace_u
        no_mask_input = model_cfg.get('no_mask_input', False)  # 🔥 encoder sees mask=1 (no mask info)
        fo_kwargs = model_cfg.get('fo_kwargs', None)  # extra kwargs for fo module
        use_mask_decoder = model_cfg.get('use_mask_decoder', False)  # enable u_o mask decoder on smoothed Z
        
        # Determine device index
        if isinstance(self.device, torch.device):
            device_idx = self.device.index if self.device.type == 'cuda' else 0
        else:
            device_idx = 0
        
        model = FBI_SSM_Joint(
            p=p,
            T=T,
            d=d,
            device=device_idx,
            learn_initial_dist=learn_initial_dist,
            hidden_scale=hidden_scale,
            fx_type=fx_type,
            fo_type=fo_type,
            is_complete_data=is_complete_data,
            encoder_version=encoder_type,
            smoother_version=smoother_type,
            decoder_type=decoder_type,
            hard_replace_u=hard_replace_u,  # 🔥 传递 hard_replace_u
            no_mask_input=no_mask_input,  # 🔥 传递 no_mask_input
            fo_kwargs=fo_kwargs,
            use_mask_decoder=use_mask_decoder,  # u_o mask decoder on smoothed Z
        )
        
        model.device = self.device
        return model.to(self.device)
    
    def _prepare_data(self, X, mask):
        """
        Prepare data for FBI format.
        
        FBI expects: (N, p, T) format
        """
        # X is already (N, p, T)
        X = np.nan_to_num(X, nan=0.0)
        return X.astype(np.float32), mask.astype(np.float32)
    
    def _create_dataloader(self, X, mask, batch_size=None, shuffle=True):
        """Create DataLoader"""
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, mask_tensor)
        bs = batch_size if batch_size is not None else self.batch_size
        dataloader = DataLoader(
            dataset, 
            batch_size=bs, 
            shuffle=shuffle
        )
        return dataloader
    
    def fit(self, X, mask, verbose=True, skip_normalize=False,
            provided_mean=None, provided_std=None):
        """
        Fit FBI model using alternating optimization.
        
        Args:
            X: np.array, shape (N, p, T) - data (NaN for missing)
            mask: np.array, shape (N, p, T) - binary mask (1=observed, 0=missing)
            verbose: bool, print progress
            skip_normalize: bool - if True, skip normalization (data already normalized)
            provided_mean: np.array, shape (p,) - use this mean instead of computing
            provided_std: np.array, shape (p,) - use this std instead of computing
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        N, p, T = X.shape
        self.shape = (p, T)
        
        # Handle normalization
        if skip_normalize:
            # Data already normalized, use identity scaler
            self.mean_scaler = np.zeros((1, p, 1), dtype=np.float32)
            self.scaler = np.ones((1, p, 1), dtype=np.float32)
            X_norm = X
            if verbose:
                print("  [FBI] Using identity scaler (data pre-normalized)")
        elif provided_mean is not None and provided_std is not None:
            # Use provided scaler
            self.mean_scaler = provided_mean.reshape(1, p, 1)
            self.scaler = provided_std.reshape(1, p, 1)
            self.scaler[self.scaler == 0] = 1.0
            X_norm = (X - self.mean_scaler) / self.scaler
            if verbose:
                print(f"  [FBI] Using provided scaler")
        else:
            # Compute normalization parameters (per-feature)
            self.mean_scaler = np.nanmean(X, axis=(0, 2), keepdims=True)  # (1, p, 1)
            self.scaler = np.nanstd(X, axis=(0, 2), keepdims=True)  # (1, p, 1)
            self.scaler[self.scaler == 0] = 1.0
            X_norm = (X - self.mean_scaler) / self.scaler
            if verbose:
                print(f"  [FBI] Computed scaler from data")
        
        # Prepare data
        X_prepared, mask_prepared = self._prepare_data(X_norm, mask)
        self._X_prepared = X_prepared  # store for Phase 2
        self._mask_prepared = mask_prepared
        
        # Create model
        self.model = self._create_model(p, T)
        
        # Get training params (need batch sizes before creating loaders)
        train_cfg = self.config.get('train', {})
        lr_fg = train_cfg.get('lr_fg', 1e-3)
        lr_huq = train_cfg.get('lr_huq', 1e-3)
        epochs_fg = train_cfg.get('epochs_fg', 1)
        epochs_huq = train_cfg.get('epochs_huq', 1)
        gen_sample_size = train_cfg.get('gen_sample_size', 1)
        alpha = train_cfg.get('alpha', 1.0)
        # syn_size 默认使用训练集大小 (None 或不设置时自动使用 N)
        syn_size = train_cfg.get('syn_size') or N
        # 分开设置各阶段 batch_size (兼容旧配置)
        batch_size_fg = train_cfg.get('batch_size_fg', train_cfg.get('batch_size', 128))
        # batch_size_huq = train_cfg.get('batch_size_huq', train_cfg.get('batch_size', 256))
        # batch_size_u = train_cfg.get('batch_size_u', train_cfg.get('batch_size', 128))
        # train_u = train_cfg.get('train_u', 'self_mask')
        drop_rate = train_cfg.get('drop_rate', 0.2)
        lambda_fo = train_cfg.get('lambda_fo', 1.0)
        lambda_fx = train_cfg.get('lambda_fx', 1.0)
        fo_loss_type = train_cfg.get('fo_loss_type', 'l2')
        
        # Create dataloader for f,g training (uses batch_size_fg)
        train_loader_fg = self._create_dataloader(X_prepared, mask_prepared, batch_size=batch_size_fg, shuffle=True)
        
        # Setup optimizers
        transition_params = list(self.model.transition_g.parameters())
        output_params = list(self.model.out_fx.parameters()) + list(self.model.out_fo.parameters())
        fg_params = transition_params + output_params + list(self.model.noise_processor.parameters())
        h_params = list(self.model.encoder_h.parameters())
        q_params = list(self.model.backward_transition_q.parameters())
        u_params = list(self.model.decoder_u.parameters())
        u_o_params = list(self.model.mask_decoder_u.parameters()) if self.model.mask_decoder_u is not None else []
        
        # optimizer_fg = optim.Adam(fg_params, lr=lr_fg)
        # optimizer_h = optim.Adam(h_params, lr=lr_huq)
        # optimizer_q = optim.Adam(q_params, lr=lr_huq)
        # optimizer_u = optim.Adam(u_params, lr=lr_huq)
        
        # ============================================================
        # Joint Training: All components (f, g, h, q, u) updated together
        # Total Loss = L_forecast(f, g, h) + λ * L_selfmask(u, q, h)
        # ============================================================
        from modules.criterion import compute_masked_l2_distance
        
        # Use a single optimizer for all parameters
        all_params = fg_params + h_params + q_params + u_params + u_o_params
        optimizer_joint = optim.Adam(all_params, lr=lr_fg)
        
        # Lambda for selfmask loss
        lambda_selfmask = train_cfg.get('lambda_selfmask', 5.0)
        lambda_d_o = train_cfg.get('lambda_d_o', 0.0)
        mask_dec_pos_weight = train_cfg.get('mask_dec_pos_weight', None)  # e.g. 4.0 for 20% obs rate
        lambda_enc = train_cfg.get('lambda_enc', 0.0)
        enc_warmup = train_cfg.get('enc_warmup', 50)
        enc_syn_batch = train_cfg.get('enc_syn_batch', None)
        if enc_syn_batch is None:
            enc_syn_batch = min(4000, N) if lambda_enc > 0.0 else 0
        enc_syn_mini_batch = train_cfg.get('enc_syn_mini_batch', None)
        if enc_syn_mini_batch is None and lambda_enc > 0.0:
            enc_syn_mini_batch = min(batch_size_fg, enc_syn_batch)
        enc_noise_scale = train_cfg.get('enc_noise_scale', 0.0)
        enc_ema_decay = train_cfg.get('enc_ema_decay', 0.999)
        is_complete_data = self.config.get('model', {}).get('is_complete_data', False)
        ema_enc = _EncoderConsistencyEMA(self.model, decay=enc_ema_decay) if lambda_enc > 0.0 else None
        
        best_loss = float('inf')
        best_state = None
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Joint Training: f, g, h, q, u (Joint-training)")
            print(f"  Epochs: {self.total_loops}")
            print(f"  lr={lr_fg}, batch_size={batch_size_fg}")
            print(f"  gen_sample_size={gen_sample_size}, drop_rate={drop_rate}")
            print(f"  lambda_selfmask={lambda_selfmask}, lambda_d_o={lambda_d_o}")
            if lambda_enc > 0.0:
                print(f"  lambda_enc={lambda_enc}, enc_warmup={enc_warmup}, "
                      f"enc_syn_batch={enc_syn_batch}, enc_syn_mini_batch={enc_syn_mini_batch}")
            print(f"{'='*60}")
        
        for epoch in range(self.total_loops):
            self.model.train()
            
            loss_total_all = []
            loss_forecast_all = []
            loss_selfmask_all = []
            loss_mask_dec_all = []
            loss_enc_all = []
            loss_select_all = []
            apply_enc = (lambda_enc > 0.0) and (epoch >= enc_warmup)
            
            for X_batch, mask_batch in train_loader_fg:
                X_batch = X_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                X_batch = torch.nan_to_num(X_batch, nan=0.0)
                
                optimizer_joint.zero_grad()
                
                # ============================================================
                # Part 1: L_forecast(f, g, h) - Forecasting Loss
                # ============================================================
                forecasting_X1, forecasting_X2, forecasting_O1, forecasting_O2, Z_f = self.model.forecasting(
                    X_batch, mask_batch, gen_sample_size=gen_sample_size, train_flag=True
                )
                
                X_true = X_batch.unsqueeze(1).expand_as(forecasting_X1)
                mask_true = mask_batch.unsqueeze(1).expand_as(forecasting_X1)
                
                loss_f1 = compute_masked_l2_distance(X_true, forecasting_X1, obs_mask=mask_true, 
                                                      obs_mask_prime=forecasting_O1, lambda_fo=lambda_fo, lambda_fx=lambda_fx, fo_loss_type=fo_loss_type)
                loss_f2 = compute_masked_l2_distance(X_true, forecasting_X2, obs_mask=mask_true, 
                                                      obs_mask_prime=forecasting_O2, lambda_fo=lambda_fo, lambda_fx=lambda_fx, fo_loss_type=fo_loss_type)
                loss_f3 = compute_masked_l2_distance(forecasting_X1, forecasting_X2, obs_mask=forecasting_O1, 
                                                      obs_mask_prime=forecasting_O2, div=mask_true, lambda_fo=lambda_fo, lambda_fx=lambda_fx, fo_loss_type=fo_loss_type)
                
                loss_forecast = loss_f1 + loss_f2 - loss_f3
                loss_forecast_all.append(loss_forecast.item())
                
                # ============================================================
                # Part 2: L_selfmask(u, q, h) - Self-Masking Imputation Loss
                # ============================================================
                if is_complete_data:
                    loss_selfmask = torch.tensor(0.0, device=self.device)
                else:
                    X_btp = X_batch.permute(0, 2, 1)  # (B, T, p)
                    mask_btp = mask_batch.permute(0, 2, 1)  # (B, T, p)
                    
                    # Generate self-mask
                    if isinstance(drop_rate, str) and drop_rate.lower() == 'random':
                        B = mask_btp.shape[0]
                        sample_drop_rates = torch.rand(B, 1, 1, device=self.device)
                        rand_matrix = torch.rand_like(mask_btp.float())
                        keep_mask = (rand_matrix > sample_drop_rates).float()
                    else:
                        rand_matrix = torch.rand_like(mask_btp.float())
                        keep_mask = (rand_matrix > drop_rate).float()
                    
                    O_input = mask_btp * keep_mask
                    X_input = X_btp * O_input
                    target_mask = mask_btp * (1 - keep_mask)
                    
                    if target_mask.sum() == 0:
                        loss_selfmask = torch.tensor(0.0, device=self.device)
                    else:
                        O_input_enc = torch.ones_like(O_input) if self.model.no_mask_input else O_input
                        Z_f_sm = self.model.encoder_h(X_input, O_input_enc)
                        Z_s1 = self.model.run_backward_pass(Z_f_sm)
                        Z_s2 = self.model.run_backward_pass(Z_f_sm)
                        
                        # Only run X-decoder when lambda_selfmask > 0 (skip if mask-decoder-only)
                        if lambda_selfmask > 0:
                            X_recon1 = self.model.decoder_u(Z_s1, X_input, O_input)
                            X_recon2 = self.model.decoder_u(Z_s2, X_input, O_input)
                            
                            loss_u1 = compute_masked_l2_distance(X_btp, X_recon1, obs_mask=mask_btp)
                            loss_u2 = compute_masked_l2_distance(X_btp, X_recon2, obs_mask=mask_btp)
                            loss_u3 = compute_masked_l2_distance(X_recon1, X_recon2, obs_mask=mask_btp)
                            
                            loss_selfmask = loss_u1 + loss_u2 - loss_u3
                        else:
                            loss_selfmask = torch.tensor(0.0, device=self.device)
                
                loss_selfmask_all.append(loss_selfmask.item())
                
                # ============================================================
                # Part 3: L_mask_dec(u_o) - Mask Decoder BCE Loss (optional)
                # ============================================================
                loss_mask_dec = torch.tensor(0.0, device=self.device)
                if lambda_d_o > 0 and self.model.use_mask_decoder and not is_complete_data:
                    if target_mask.sum() > 0:
                        # Reuse out_fo for mask prediction (Z_s → probs via sigmoid inside out_fo)
                        prob_j1 = self.model.out_fo(Z_s1).clamp(1e-7, 1 - 1e-7)  # (B, T, p)
                        prob_j2 = self.model.out_fo(Z_s2).clamp(1e-7, 1 - 1e-7)  # (B, T, p)
                        if mask_dec_pos_weight is not None:
                            pw = torch.tensor(mask_dec_pos_weight, device=self.device)
                            bce_j1 = -(pw * mask_btp * torch.log(prob_j1)
                                       + (1 - mask_btp) * torch.log(1 - prob_j1)).mean()
                            bce_j2 = -(pw * mask_btp * torch.log(prob_j2)
                                       + (1 - mask_btp) * torch.log(1 - prob_j2)).mean()
                        else:
                            bce_j1 = torch.nn.functional.binary_cross_entropy(prob_j1, mask_btp)
                            bce_j2 = torch.nn.functional.binary_cross_entropy(prob_j2, mask_btp)
                        loss_mask_dec = 0.5 * (bce_j1 + bce_j2)
                
                loss_mask_dec_all.append(loss_mask_dec.item())
                
                # ============================================================
                # Total Loss
                # ============================================================
                loss_total = loss_forecast + lambda_selfmask * loss_selfmask + lambda_d_o * loss_mask_dec
                loss_select = loss_total.detach()
                loss_enc = torch.tensor(0.0, device=self.device)
                if apply_enc:
                    loss_enc = _compute_encoder_consistency_loss(
                        self.model, ema_enc, enc_syn_batch, enc_syn_mini_batch,
                        enc_noise_scale, self.device
                    )
                    loss_total = loss_total + lambda_enc * loss_enc

                loss_total.backward()
                optimizer_joint.step()
                if ema_enc is not None:
                    ema_enc.update(self.model)
                loss_total_all.append(loss_total.item())
                loss_select_all.append(loss_select.item())
                loss_enc_all.append(loss_enc.item())
            
            # Track best model
            mean_loss = np.mean(loss_select_all) if loss_select_all else np.mean(loss_total_all)
            if mean_loss < best_loss and epoch >= self.total_loops // 4:
                best_loss = mean_loss
                best_state = copy.deepcopy(self.model.state_dict())
            
            if verbose:
                mean_fore = np.mean(loss_forecast_all)
                mean_imp = np.mean(loss_selfmask_all)
                mean_msk = np.mean(loss_mask_dec_all)
                mean_enc = np.mean(loss_enc_all)
                best_str = f"★{best_loss:.3f}" if best_state is not None else ""
                progress = (epoch + 1) / self.total_loops * 100
                msk_str = f" msk={mean_msk:.3f}" if lambda_d_o > 0 else ""
                enc_str = f" enc={mean_enc:.3f}" if apply_enc else ""
                print(f"\rJoint: [{epoch+1:3d}/{self.total_loops}] {progress:5.1f}% | "
                      f"total={mean_loss:.3f} (fore={mean_fore:.3f} imp={mean_imp:.3f}{msk_str}{enc_str}) {best_str}", 
                      end="", flush=True)
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print()
                print(f"Joint training finished. Best total loss: {best_loss:.4f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete! Best loss: {best_loss:.4f}")
            print(f"{'='*60}\n")
        
        self.trained = True
        
    def _train_hq_phase(self, optimizer_h, optimizer_q, epochs_huq, syn_size, batch_size_huq):
        """Train h, q only (NOT u). Returns: model, total_loss, loss_h, loss_q, and placeholders"""
        from modules.criterion import compute_masked_l2_distance
        from loader.transforms import convert_data
        import numpy as np
        
        # Freeze f, g
        for name, param in self.model.named_parameters():
            if 'noise_processor' in name or 'out_fx' in name or 'out_fo' in name or 'transition_g' in name:
                param.requires_grad = False
            elif 'encoder_h' in name or 'backward_transition_q' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # freeze decoder_u
        
        # Generate synthetic data
        self.model.eval()
        with torch.no_grad():
            X_syn, O_syn, Z_syn = self.model.generate_synthetic_data_from_fg(synthetic_size=syn_size)
        
        train_loader_huq = convert_data(X_syn, O_syn, batch_size_huq, X_complete=Z_syn)
        
        self.model.train()
        self.model.out_fx.eval()
        self.model.out_fo.eval()
        self.model.noise_processor.eval()
        self.model.transition_g.eval()
        self.model.encoder_h.train()
        
        loss_h_all = []
        loss_q_all = []
        
        for epoch in range(epochs_huq):
            for X_syn_batch, O_syn_batch, Z_syn_batch in train_loader_huq:
                X_syn_batch = X_syn_batch.to(self.device)
                O_syn_batch = O_syn_batch.to(self.device)
                Z_syn_batch = Z_syn_batch.to(self.device)
                
                # Train h
                optimizer_h.zero_grad()
                O_syn_enc = torch.ones_like(O_syn_batch) if self.model.no_mask_input else O_syn_batch
                Z_f1 = self.model.encoder_h(X_syn_batch, O_syn_enc)
                Z_f2 = self.model.encoder_h(X_syn_batch, O_syn_enc)
                loss_h1 = compute_masked_l2_distance(Z_syn_batch, Z_f1)
                loss_h2 = compute_masked_l2_distance(Z_syn_batch, Z_f2)
                loss_h3 = compute_masked_l2_distance(Z_f1, Z_f2)
                loss_h = loss_h1 + loss_h2 - loss_h3
                loss_h.backward()
                optimizer_h.step()
                loss_h_all.append(loss_h.item())
                
                # Train q
                optimizer_q.zero_grad()
                with torch.no_grad():
                    O_syn_enc = torch.ones_like(O_syn_batch) if self.model.no_mask_input else O_syn_batch
                    Z_f1 = self.model.encoder_h(X_syn_batch, O_syn_enc)
                    Z_f2 = self.model.encoder_h(X_syn_batch, O_syn_enc)
                # FBI_SSM_Joint uses run_backward_pass (inference-style) for training as well
                Z_s1 = self.model.run_backward_pass(Z_f1)
                Z_s2 = self.model.run_backward_pass(Z_f2)
                loss_q1 = compute_masked_l2_distance(Z_syn_batch, Z_s1)
                loss_q2 = compute_masked_l2_distance(Z_syn_batch, Z_s2)
                loss_q3 = compute_masked_l2_distance(Z_s1, Z_s2)
                loss_q = loss_q1 + loss_q2 - loss_q3
                loss_q.backward()
                optimizer_q.step()
                loss_q_all.append(loss_q.item())
        
        mean_loss_h = np.mean(loss_h_all) if loss_h_all else 0.0
        mean_loss_q = np.mean(loss_q_all) if loss_q_all else 0.0
        total_loss = mean_loss_h + mean_loss_q
        
        return self.model, total_loss, mean_loss_h, mean_loss_q, 0.0, 0.0, 0.0, 0.0
    
    def _train_u_phase(self, optimizer_u, epochs_u, train_u, train_loader, drop_rate, verbose=False):
        """Train decoder u only (Phase 2). Returns: model, best_loss, best_state"""
        from modules.criterion import compute_masked_l2_distance
        import numpy as np
        
        # Freeze f, g, h, q - only train u
        for name, param in self.model.named_parameters():
            if 'decoder_u' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.model.eval()  # set all to eval
        self.model.decoder_u.train()  # only u in train mode
        
        best_loss_u = float('inf')
        best_state_u = None
        
        for epoch in range(epochs_u):
            loss_u_all = []
            loss_u1_all = []
            loss_u2_all = []
            loss_u3_all = []
            
            for X_batch, mask_batch in train_loader:
                X_batch = X_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                X_batch = torch.nan_to_num(X_batch, nan=0.0)
                
                # Permute to (B, T, p) for model
                X_perm = X_batch.permute(0, 2, 1)
                mask_perm = mask_batch.permute(0, 2, 1)
                
                # Create self-mask
                rand_matrix = torch.rand_like(mask_perm)
                keep_mask = (rand_matrix > drop_rate).float()
                O_input = mask_perm * keep_mask
                X_input = X_perm * O_input
                target_mask = mask_perm * (1 - keep_mask)
                
                if target_mask.sum() == 0:
                    continue
                
                optimizer_u.zero_grad()
                
                with torch.no_grad():
                    O_input_enc = torch.ones_like(O_input) if self.model.no_mask_input else O_input
                    Z_f = self.model.encoder_h(X_input, O_input_enc)
                    Z_s = self.model.run_backward_pass(Z_f)
                
                X_recon1 = self.model.decoder_u(Z_s, X_input, O_input)
                X_recon2 = self.model.decoder_u(Z_s, X_input, O_input)
                
                loss_u1 = compute_masked_l2_distance(X_perm, X_recon1, obs_mask=mask_perm)
                loss_u2 = compute_masked_l2_distance(X_perm, X_recon2, obs_mask=mask_perm)
                loss_u3 = compute_masked_l2_distance(X_recon1, X_recon2, obs_mask=mask_perm)
                loss_u = loss_u1 + loss_u2 - loss_u3
                
                loss_u.backward()
                optimizer_u.step()
                
                loss_u_all.append(loss_u.item())
                loss_u1_all.append(loss_u1.item())
                loss_u2_all.append(loss_u2.item())
                loss_u3_all.append(loss_u3.item())
            
            # Compute epoch mean
            mean_u = np.mean(loss_u_all) if loss_u_all else 0.0
            mean_u1 = np.mean(loss_u1_all) if loss_u1_all else 0.0
            mean_u2 = np.mean(loss_u2_all) if loss_u2_all else 0.0
            mean_u3 = np.mean(loss_u3_all) if loss_u3_all else 0.0
            
            # Track best (after warmup)
            if mean_u < best_loss_u and epoch >= epochs_u // 2:
                best_loss_u = mean_u
                best_state_u = copy.deepcopy(self.model.state_dict())
            
            if verbose:
                best_str = f"★{best_loss_u:.3f}" if best_state_u is not None else ""
                progress = (epoch + 1) / epochs_u * 100
                print(f"\rPhase 2: [{epoch+1:3d}/{epochs_u}] {progress:5.1f}% | "
                      f"u={mean_u:.3f} (u1={mean_u1:.3f} u2={mean_u2:.3f} u3={mean_u3:.3f}) {best_str}", 
                      end="", flush=True)
        
        if verbose:
            print()  # newline after progress
        
        return self.model, best_loss_u, best_state_u
    
    def _train_fg_phase(self, train_loader, optimizer_fg, epochs_fg,
                         gen_sample_size, alpha, lambda_fo, verbose=False, fo_loss_type='l2'):
        """Train f, g (forward model). Returns: model, total_loss, loss1, loss2, energy_loss"""
        from modules.criterion import compute_masked_l2_distance
        
        # Freeze h, u, q
        for name, param in self.model.named_parameters():
            if 'encoder_h' in name or 'backward_transition_q' in name or 'decoder_u' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.model.train()
        
        total_loss_history = []
        loss1_history = []
        loss2_history = []
        energy_loss_history = []
        
        n_batches = len(train_loader)
        total_iters = epochs_fg * n_batches
        current_iter = 0
        
        for epoch in range(epochs_fg):
            for X_batch, mask_batch in train_loader:
                X_batch = X_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                X_batch = torch.nan_to_num(X_batch, nan=0.0)
                
                optimizer_fg.zero_grad()
                
                # Forward pass
                fx1, fx2, fo1, fo2, Z_f = self.model.forecasting(
                    X_batch, mask_batch, gen_sample_size=gen_sample_size, train_flag=True
                )
                
                X_true = X_batch.unsqueeze(1).expand_as(fx1)
                mask_true = mask_batch.unsqueeze(1).expand_as(fx1)
                
                # Energy score loss
                loss1 = compute_masked_l2_distance(X_true, fx1, obs_mask=mask_true,
                                                    obs_mask_prime=fo1, lambda_fo=lambda_fo, fo_loss_type=fo_loss_type)
                loss2 = compute_masked_l2_distance(X_true, fx2, obs_mask=mask_true,
                                                    obs_mask_prime=fo2, lambda_fo=lambda_fo, fo_loss_type=fo_loss_type)
                loss3 = compute_masked_l2_distance(fx1, fx2, obs_mask=fo1,
                                                    obs_mask_prime=fo2, div=mask_true, lambda_fo=lambda_fo, fo_loss_type=fo_loss_type)
                
                loss = loss1 + loss2 - loss3
                loss.backward()
                optimizer_fg.step()
                
                total_loss_history.append(loss.item())
                loss1_history.append(loss1.item())
                loss2_history.append(loss2.item())
                energy_loss_history.append(loss3.item())
                
                current_iter += 1
                if verbose:
                    progress = current_iter / total_iters * 100
                    print(f"\r  Phase 1 (f,g): [{current_iter:4d}/{total_iters}] {progress:5.1f}% | "
                          f"fg={loss.item():.3f} (f1={loss1.item():.3f} f2={loss2.item():.3f} e={loss3.item():.3f})",
                          end="", flush=True)
        
        if verbose:
            print()  # newline after progress
        
        import numpy as np
        mean_total = np.mean(total_loss_history) if total_loss_history else 0.0
        mean_loss1 = np.mean(loss1_history) if loss1_history else 0.0
        mean_loss2 = np.mean(loss2_history) if loss2_history else 0.0
        mean_energy = np.mean(energy_loss_history) if energy_loss_history else 0.0
        
        return self.model, mean_total, mean_loss1, mean_loss2, mean_energy
    
    def impute(self, X, mask, n_samples=None):
        """
        Impute missing values (return mean imputation).
        
        Args:
            X: np.array, shape (N, p, T)
            mask: np.array, shape (N, p, T)
            n_samples: int, number of samples to average
            
        Returns:
            np.array, shape (N, p, T) - imputed data
        """
        samples = self.impute_samples(X, mask, n_samples)
        return samples.mean(axis=1)
    
    def impute_samples(self, X, mask, n_samples=None, batch_size=10, samples_per_forward=100):
        """
        Return multiple imputation samples.
        
        Args:
            X: np.array, shape (N, p, T)
            mask: np.array, shape (N, p, T)
            n_samples: int, number of samples
            batch_size: int, number of data samples to process at once (default: 10)
            samples_per_forward: int, number of imputation samples to generate per forward pass (default: 100)
                               Uses model's gen_sample_size for parallel sample generation
            
        Returns:
            np.array, shape (N, m, p, T) - imputation samples
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before imputation")
        
        n_samples = n_samples or self.n_samples
        N, p, T = X.shape
        
        # Normalize
        X_norm = (X - self.mean_scaler) / self.scaler
        
        # Prepare data
        X_prepared, mask_prepared = self._prepare_data(X_norm, mask)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_prepared, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(mask_prepared, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        
        all_samples = []
        
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = min(batch_size, N)
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_tensor[start_idx:end_idx]
                mask_batch = mask_tensor[start_idx:end_idx]
                actual_batch_size = end_idx - start_idx
                
                batch_samples = []
                
                # Generate samples in chunks using gen_sample_size for parallelization
                remaining_samples = n_samples
                while remaining_samples > 0:
                    chunk_size = min(samples_per_forward, remaining_samples)
                    
                    # Generate chunk_size samples in one forward pass using gen_sample_size
                    _, _, _, _, smooth_X1, smooth_X2, _, _ = self.model.forward(
                        X_batch, mask_batch, gen_sample_size=chunk_size
                    )
                    # smooth_X1: (batch, chunk_size, p, T)
                    batch_samples.append(smooth_X1.cpu().numpy())
                    
                    remaining_samples -= chunk_size
                
                # Concatenate chunks: (batch, n_samples, p, T)
                batch_samples = np.concatenate(batch_samples, axis=1)
                all_samples.append(batch_samples)
        
        # Concatenate all batches
        all_samples = np.concatenate(all_samples, axis=0)  # (N, n_samples, p, T)
        
        # Denormalize: scaler is (1, p, 1), need to broadcast with (N, n_samples, p, T)
        # Add axis for n_samples dimension: (1, p, 1) -> (1, 1, p, 1)
        all_samples = all_samples * self.scaler[:, np.newaxis, :, :] + \
                      self.mean_scaler[:, np.newaxis, :, :]
        
        # Keep original observed values
        mask_expanded = mask[:, np.newaxis, :, :]  # (N, 1, p, T)
        X_expanded = X[:, np.newaxis, :, :]  # (N, 1, p, T)
        all_samples = np.where(mask_expanded == 1, X_expanded, all_samples)
        
        return all_samples
    
    def forecast(self, X, mask, n_samples=None):
        """
        Forecast (in-sample) and return mean forecast.
        
        Args:
            X: np.array, shape (N, p, T)
            mask: np.array, shape (N, p, T)
            n_samples: int, number of samples to average
            
        Returns:
            np.array, shape (N, p, T) - forecasted data
        """
        samples = self.forecast_samples(X, mask, n_samples)
        return samples.mean(axis=1)
    
    def forecast_samples(self, X, mask, n_samples=None, batch_size=64, samples_per_forward=10):
        """
        Return multiple forecast samples using the (f, g) model.
        This is in-sample forecasting: uses observed data to encode latent Z,
        then uses transition g and output f to forecast X.
        
        Args:
            X: np.array, shape (N, p, T)
            mask: np.array, shape (N, p, T)
            n_samples: int, number of samples
            batch_size: int, number of data samples to process at once (default: 64)
            samples_per_forward: int, number of forecast samples to generate per forward pass (default: 10)
            
        Returns:
            np.array, shape (N, m, p, T) - forecast samples
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before forecasting")
        
        n_samples = n_samples or self.n_samples
        N, p, T = X.shape
        
        # Normalize
        X_norm = (X - self.mean_scaler) / self.scaler
        
        # Prepare data
        X_prepared, mask_prepared = self._prepare_data(X_norm, mask)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_prepared, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(mask_prepared, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        
        all_samples = []
        
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = min(batch_size, N)
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_tensor[start_idx:end_idx]
                mask_batch = mask_tensor[start_idx:end_idx]
                
                batch_samples = []
                
                # Generate samples in chunks using gen_sample_size for parallelization
                remaining_samples = n_samples
                while remaining_samples > 0:
                    chunk_size = min(samples_per_forward, remaining_samples)
                    
                    # Run forecasting with gen_sample_size for parallel sample generation
                    fx1, fx2, fo1, fo2, Z_f = self.model.forecasting(
                        X_batch, mask_batch, gen_sample_size=chunk_size, train_flag=True
                    )
                    # fx1: (batch, chunk_size, p, T)
                    batch_samples.append(fx1.cpu().numpy())
                    
                    remaining_samples -= chunk_size
                
                # Concatenate chunks: (batch, n_samples, p, T)
                batch_samples = np.concatenate(batch_samples, axis=1)
                all_samples.append(batch_samples)
        
        # Concatenate all batches
        all_samples = np.concatenate(all_samples, axis=0)  # (N, n_samples, p, T)
        
        # Denormalize
        all_samples = all_samples * self.scaler[:, np.newaxis, :, :] + \
                      self.mean_scaler[:, np.newaxis, :, :]
        
        # Note: For forecasting, we do NOT replace with observed values
        # because forecasting is predicting the full signal, not imputing missing values
        
        return all_samples
    
    def save(self, path):
        """Save model and normalization parameters"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'mean_scaler': self.mean_scaler,
            'scaler': self.scaler,
            'shape': self.shape,
        }, path)
    
    def load(self, path):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.mean_scaler = checkpoint['mean_scaler']
        self.scaler = checkpoint['scaler']
        self.shape = checkpoint['shape']
        
        p, T = self.shape
        self.model = self._create_model(p, T)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True


def fbi_impute(X_train, mask_train, X_test, mask_test,
               total_loops=100, batch_size=256, n_samples=10,
               seed=None, verbose=True):
    """
    Functional interface for FBI imputation.
    
    Args:
        X_train: np.array, shape (N_train, p, T) - training data
        mask_train: np.array, shape (N_train, p, T) - training mask
        X_test: np.array, shape (N_test, p, T) - test data
        mask_test: np.array, shape (N_test, p, T) - test mask
        total_loops: int, total alternating training loops
        batch_size: int
        n_samples: int, imputation samples
        seed: int
        verbose: bool
        
    Returns:
        np.array, shape (N_test, p, T) - imputed test data
    """
    imputer = FBIImputer(
        total_loops=total_loops,
        batch_size=batch_size,
        n_samples=n_samples,
        seed=seed
    )
    
    # Fit on training data
    imputer.fit(X_train, mask_train, verbose=verbose)
    
    # Impute test data
    imputed = imputer.impute(X_test, mask_test, n_samples=n_samples)
    
    return imputed


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    torch.manual_seed(42)
    
    N_train, N_test, p, T = 100, 20, 10, 20
    X_train = np.random.randn(N_train, p, T).astype(np.float32)
    X_test = np.random.randn(N_test, p, T).astype(np.float32)
    
    # Create random masks (20% missing)
    mask_train = (np.random.rand(N_train, p, T) > 0.2).astype(np.float32)
    mask_test = (np.random.rand(N_test, p, T) > 0.2).astype(np.float32)
    
    # Apply mask
    X_train_missing = X_train.copy()
    X_train_missing[mask_train == 0] = np.nan
    
    X_test_missing = X_test.copy()
    X_test_missing[mask_test == 0] = np.nan
    
    # Impute
    imputed = fbi_impute(
        X_train_missing, mask_train,
        X_test_missing, mask_test,
        total_loops=10,
        n_samples=5,
        verbose=True
    )
    
    # Compute MAE on missing values
    mae = np.mean(np.abs(imputed - X_test)[mask_test == 0])
    print(f"Test MAE on missing values: {mae:.4f}")
