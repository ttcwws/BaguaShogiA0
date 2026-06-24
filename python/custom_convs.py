import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalNeighborConv(nn.Module):
    """Convolution over only 4 diagonal neighbors: TL, TR, BL, BR."""

    def __init__(self, num_in, num_out, bias=True):
        super().__init__()
        # A single 3x3 kernel where only 4 corners are trainable.
        self.kernel = nn.Parameter(torch.empty(num_out, num_in, 3, 3))
        self.bias = nn.Parameter(torch.empty(num_out)) if bias else None
        mask = torch.zeros(num_out, num_in, 3, 3)
        mask[:, :, 0, 0] = 1.0
        mask[:, :, 0, 2] = 1.0
        mask[:, :, 2, 0] = 1.0
        mask[:, :, 2, 2] = 1.0
        self.register_buffer("corner_mask", mask)
        self.kernel.register_hook(lambda g: g * self.corner_mask)
        bound = (1.0 / (num_in * 4)) ** 0.5
        nn.init.zeros_(self.kernel)
        nn.init.uniform_(self.kernel[:, :, 0, 0], -bound, bound)
        nn.init.uniform_(self.kernel[:, :, 0, 2], -bound, bound)
        nn.init.uniform_(self.kernel[:, :, 2, 0], -bound, bound)
        nn.init.uniform_(self.kernel[:, :, 2, 2], -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv2d(x, self.kernel, bias=self.bias, padding=1)

    def numel(self):
        # Effective trainable params: 4 corners per (out,in) + optional bias.
        corner_params = int(self.corner_mask.sum().item())
        bias_params = 0 if self.bias is None else self.bias.numel()
        return corner_params + bias_params

class ShogiConv(nn.Module):
    """33-param cross kernel via optimized Conv2d branches.

    Uses:
    - horizontal 1x17 branch (left8 + center + right8)
    - vertical 17x1 branch (up8 + center + down8)
    - subtract center 1x1 once to avoid double counting
    """

    def __init__(self, num_in, num_out, bias=True):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.w_left = nn.Parameter(torch.empty(num_out, num_in, 8))
        self.w_right = nn.Parameter(torch.empty(num_out, num_in, 8))
        self.w_up = nn.Parameter(torch.empty(num_out, num_in, 8))
        self.w_down = nn.Parameter(torch.empty(num_out, num_in, 8))
        self.w_center = nn.Parameter(torch.empty(num_out, num_in))
        self.bias = nn.Parameter(torch.empty(num_out)) if bias else None
        self.register_buffer("zero_center", torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        bound = (1.0 / (self.num_in * 33)) ** 0.5
        nn.init.uniform_(self.w_left, -bound, bound)
        nn.init.uniform_(self.w_right, -bound, bound)
        nn.init.uniform_(self.w_up, -bound, bound)
        nn.init.uniform_(self.w_down, -bound, bound)
        nn.init.uniform_(self.w_center, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        w_h = torch.cat(
            [self.w_left, self.w_center.unsqueeze(-1), self.w_right], dim=2
        ).unsqueeze(2)
        zero_center = self.zero_center.expand(self.num_out, self.num_in, 1)
        w_v = torch.cat(
            [self.w_up, zero_center, self.w_down], dim=2
        ).unsqueeze(3)

        # Fuse bias into one branch to avoid an extra elementwise add kernel.
        y = F.conv2d(x, w_h, bias=self.bias, padding=(0, 8))
        y.add_(F.conv2d(x, w_v, bias=None, padding=(8, 0)))
        return y

class ShogiConvNew(nn.Module):
    """保留之前的精简/优化版本为 `ShogiConvNew`（仅 `w_h`/`w_v`，`w_v` 中心为 0 且无梯度）。"""

    def __init__(self, num_in, num_out, bias=True):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        # only fused kernels as parameters
        self.w_h = nn.Parameter(torch.empty(num_out, num_in, 1, 17))
        self.w_v = nn.Parameter(torch.empty(num_out, num_in, 17, 1))
        self.bias = nn.Parameter(torch.empty(num_out)) if bias else None

        # init
        self.reset_parameters()

        # hook to zero gradients at center of w_v
        def zero_center_grad(g):
            g[:, :, 8, 0] = 0.0
            return g
        self.w_v.register_hook(zero_center_grad)

    def reset_parameters(self):
        bound = (1.0 / (self.num_in * 33)) ** 0.5
        nn.init.uniform_(self.w_h, -bound, bound)
        nn.init.uniform_(self.w_v, -bound, bound)
        # ensure center is zero
        with torch.no_grad():
            self.w_v[:, :, 8, 0].zero_()
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = F.conv2d(x, self.w_h, bias=self.bias, padding=(0, 8))
        y.add_(F.conv2d(x, self.w_v, bias=None, padding=(8, 0)))
        return y
    
class OmniConv(nn.Module):
    """Combine ShogiConv and DiagonalNeighborConv with weighted averaging.

    Weighted by multiplication counts:
    - Diagonal branch weight: 4
    - ShogiConv branch weight: 18
    Final output = (4 * y_diag + 18 * y_shogi) / 22 + shared_bias
    """

    def __init__(self, num_in, num_out, bias=True):
        super().__init__()
        self.shogi = ShogiConv(num_in, num_out, bias=False)
        self.diag = DiagonalNeighborConv(num_in, num_out, bias=False)
        self.shared_bias = nn.Parameter(torch.zeros(num_out)) if bias else None

    def forward(self, x):
        y = self.shogi(x)
        y.add_(self.diag(x))
        if self.shared_bias is not None:
            y.add_(self.shared_bias.view(1, -1, 1, 1))
        return y

    def numel(self):
        shogi_params = sum(p.numel() for p in self.shogi.parameters())
        if hasattr(self.diag, "numel") and callable(self.diag.numel):
            diag_params = self.diag.numel()
        else:
            diag_params = sum(p.numel() for p in self.diag.parameters())
        shared_bias_params = 0 if self.shared_bias is None else self.shared_bias.numel()
        return shogi_params + diag_params + shared_bias_params

