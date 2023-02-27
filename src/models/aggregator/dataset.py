import numpy as np
import torch
from torch.utils import data


class AggregatorDataset(data.Dataset):
    def __init__(self, data, labels, noisey_samples=False):
        self.data = data
        self.labels = labels
        self.noisey_samples = noisey_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # sample[:, -1] = sample[:, -1] ** 2
        # sort by retreival score
        sorted_idxs = sample[:, -1].argsort()
        sorted_sample = sample[sorted_idxs]
        if self.noisey_samples:
            sorted_sample += np.random.normal(0, 0.1, size=(5, 4))
            sorted_sample = np.clip(sorted_sample, 0.0001, 0.9999)
        sample = torch.tensor(sorted_sample.flatten())
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
