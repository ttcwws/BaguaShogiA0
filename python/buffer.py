import torch
import os
import numpy as np
class Buffer:
    def __init__(self, max_size):
        self.color = torch.tensor([])
        self.overall = torch.tensor([])
        self.policy = torch.tensor([])
        self.value = torch.tensor([])
        self.max_size = max_size
    def append(self, color, overall, policy, value):
        if color.shape[0] > self.max_size:
            color = color[-self.max_size:]
            overall = overall[-self.max_size:]
            policy = policy[-self.max_size:]
            value = value[-self.max_size:]
        new_size = self.color.shape[0] + color.shape[0]
        if new_size > self.max_size:
            self.color = self.color[color.shape[0] - self.max_size:]
            self.overall = self.overall[overall.shape[0] - self.max_size:]
            self.policy = self.policy[policy.shape[0] - self.max_size:]
            self.value = self.value[value.shape[0] - self.max_size:]
        self.color = torch.cat((self.color, color), dim=0)
        self.overall = torch.cat((self.overall, overall), dim=0)
        self.policy = torch.cat((self.policy, policy), dim=0)
        self.value = torch.cat((self.value, value), dim=0)
    def append_folder(self, path):
        if os.path.exists(path + 'color.pt'):
            color = torch.load(path + 'color.pt')
            overall = torch.load(path + 'overall.pt')
            policy = torch.load(path + 'policy.pt')
            value = torch.load(path + 'value.pt')
            self.append(color, overall, policy, value)
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.color, path + 'color.pt')
        torch.save(self.overall, path + 'overall.pt')
        torch.save(self.policy, path + 'policy.pt')
        torch.save(self.value, path + 'value.pt')
    def __len__(self):
        return self.color.shape[0]
    def choice(self, size):
        idx = np.random.choice(len(self), size, replace=False)
        return self[idx]
    def __getitem__(self, key):
        return self.color[key], self.overall[key], self.policy[key], self.value[key]