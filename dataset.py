import numpy as np
import torch
from torch.utils.data import Dataset

class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        past = torch.from_numpy(self.X[idx])
        future = torch.from_numpy(self.y[idx])
        return past, future