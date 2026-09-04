import torch
from torch.utils.data import DataLoader, TensorDataset


def convert_data(X, mask, batch_size=128, X_complete=None, shuffle=True):
    """Create a minimal DataLoader used by FBI training utilities."""
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32)

    if X_complete is None:
        dataset = TensorDataset(X_tensor, mask_tensor)
    else:
        X_complete_tensor = torch.as_tensor(X_complete, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, mask_tensor, X_complete_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
