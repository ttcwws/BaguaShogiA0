import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from custom_convs import DiagonalNeighborConv, ShogiConv
policy_size = 4790
policy_filter = 150
input_area = 9 * 9
p_c, p_h, p_w = [], [], []
with open('valid_policy.txt', 'r') as f:
    for line in f:
        c, h, w = map(int, line.split())
        p_c.append(c)
        p_h.append(h)
        p_w.append(w)
assert len(p_c) == policy_size
p_t = torch.tensor(np.array([p_c, p_h, p_w]), dtype=torch.int32)
def mish(x):
    return x * torch.tanh(F.softplus(x))
class Mish(nn.Module):
    def forward(self, x):
        return mish(x)
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = mish(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x += residual
        x = mish(self.bn2(x))
        return x
class NestedBottleneck(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        num_middle = num_hidden // 2
        self.conv1 = nn.Conv2d(num_hidden, num_middle, kernel_size=1, padding="same")
        self.bn1 = nn.BatchNorm2d(num_middle)
        self.rn1 = ResBlock(num_middle)
        self.rn2 = ResBlock(num_middle)
        self.conv2 = nn.Conv2d(num_middle, num_hidden, kernel_size=1, padding="same")
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = mish(self.bn1(self.conv1(x)))
        x = self.rn1(x)
        x = self.rn2(x)
        x = self.conv2(x)
        x += residual
        x = mish(self.bn2(x))
        return x
class PoolingBias(nn.Module):
    def __init__(self, num_out, num_pool):
        super().__init__()
        self.linear = nn.Linear(2 * num_pool, num_out)
    def forward(self, x, x2):
        x = x.flatten(start_dim=2)
        layer_mean = torch.mean(x, dim=2, keepdim=False)
        layer_max, _ = torch.max(x, dim=2, keepdim=False)
        pool = torch.cat((layer_mean, layer_max), dim=1)
        x = self.linear(pool).unsqueeze(-1).unsqueeze(-1)
        return x + x2

class HeadPooling(nn.Module):
    def __init__(self, num_out, num_pool):
        super().__init__()
        self.linear = nn.Linear(2 * num_pool, num_out)
    def forward(self, x):
        x = x.flatten(start_dim=2)
        layer_mean = torch.mean(x, dim=2, keepdim=False)
        layer_max, _ = torch.max(x, dim=2, keepdim=False)
        pool = torch.cat((layer_mean, layer_max), dim=1)
        x = self.linear(pool)
        return x

class PoolingBlock(nn.Module):
    def __init__(self, num_in, num_out, num_pool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_out - num_pool)
        self.pooling = PoolingBias(num_out - num_pool, num_pool)
        self.conv_p1 = nn.Conv2d(num_in, num_pool, kernel_size=3, padding="same",bias=False)
        self.conv_p2 = nn.Conv2d(num_in, num_out - num_pool, kernel_size=3, padding="same",bias=False)
        self.conv2 = nn.Conv2d(num_out - num_pool, num_out, kernel_size=3, padding="same",bias=False)
        self.bn2 = nn.BatchNorm2d(num_out)
        
    def forward(self, x):
        residual = x
        #pooling
        x2 = self.conv_p2(x)
        x = self.conv_p1(x)
        x = self.pooling(x, x2)

        x = mish(self.bn1(x))
        x = self.conv2(x)
        x += residual
        x = mish(self.bn2(x))
        return x
    
class PolicyHead(nn.Module):
    def __init__(self, num_hidden, policy_num):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, policy_filter, kernel_size=3, padding="same",bias=False)
        self.bn1 = nn.BatchNorm2d(policy_filter)
        self.conv2 = nn.Conv2d(num_hidden, policy_num, kernel_size=1, padding="same")
        self.bn2 = nn.BatchNorm2d(policy_num)
        self.pooling = HeadPooling(policy_filter, policy_num)
    def forward(self, x):
        b = mish(self.bn2(self.conv2(x)))
        b = self.pooling(b).unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(self.conv1(x) + b)
        return x

class ResNet(nn.Module):
    def __init__(self, input_channels_bin, input_channels_overall, num_resBlocks, num_hidden, policy_num, value_num):
        super().__init__()
        self.register_buffer('p_t', p_t.to(torch.int32))
        self.value_num = value_num
        self.startBin = nn.Conv2d(input_channels_bin, num_hidden, kernel_size=3, padding="same",bias=False)
        self.startOverall = nn.Linear(input_channels_overall, num_hidden)
        self.startBlock = nn.Sequential(
            nn.BatchNorm2d(num_hidden),
            Mish()
        )
        self.backBone = nn.ModuleList(
            [NestedBottleneck(num_hidden) for i in range(num_resBlocks)]
        )
        # self.policyHead = PolicyHead(num_hidden)
        value_ln = int((2 * value_num) ** 0.5)
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, self.value_num, kernel_size=1, padding="same"),
            nn.BatchNorm2d(self.value_num),
            Mish(),
            HeadPooling(value_ln, self.value_num),
            Mish(),
            nn.Linear(value_ln, 1),
            nn.Tanh()
        )
        self.policy_head = PolicyHead(num_hidden, policy_num)
    def forward(self, binary, overall):
        x_bin = self.startBin(binary)
        x_overall = self.startOverall(overall).unsqueeze(-1).unsqueeze(-1)
        x = x_bin + x_overall
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policy_head(x)[:, self.p_t[0], self.p_t[1], self.p_t[2]]
        policy -= policy.max(dim=1, keepdim=True)[0].detach()
        value = self.valueHead(x)
        return policy, value