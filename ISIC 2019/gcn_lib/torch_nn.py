# Path: gcn_lib/torch_nn.py
import torch
from torch import nn, Tensor
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d

def act_layer(act: str, inplace: bool = False, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    act = act.lower()
    if act == 'relu': return nn.ReLU(inplace)
    if act == 'leakyrelu': return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu': return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu': return nn.GELU()
    if act == 'hswish': return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation layer [{act}] is not found')

def norm_layer(norm: str, nc: int) -> nn.Module:
    norm = norm.lower()
    if norm == 'batch': return nn.BatchNorm2d(nc, affine=True)
    if norm == 'instance': return nn.InstanceNorm2d(nc, affine=False)
    raise NotImplementedError(f'normalization layer [{norm}] is not found')

class BasicConv(Seq):
    def __init__(self, channels: list[int], act: str = 'relu', norm: str = None, bias: bool = True, drop: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none': m.append(norm_layer(norm, channels[i]))
            if act is not None and act.lower() != 'none': m.append(act_layer(act))
            if drop > 0: m.append(nn.Dropout2d(drop))
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def batched_index_select(x: Tensor, idx: Tensor) -> Tensor:
    """
    Gathers features from a batch of tensors using indices, handling cases
    where the number of source vertices and target vertices may differ.
    """
    batch_size, num_dims, num_vertices_x, _ = x.shape
    _, num_vertices_idx, k = idx.shape

    # --- THIS IS THE KEY CHANGE ---
    # The base index should be scaled by the number of vertices in the tensor
    # we are indexing from (x), not the number of vertices in the index tensor (idx).
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_x
    
    # The index tensor 'idx' might be for a different number of vertices,
    # so we use its shape to correctly reshape the output.
    idx = (idx + idx_base).contiguous().view(-1)
    
    x = x.transpose(2, 1)
    # Flatten x to (B * N_x, C) to prepare for indexing
    feature = x.contiguous().view(batch_size * num_vertices_x, -1)[idx, :]
    
    # Reshape the output to match the shape of the index tensor 'idx'
    feature = feature.view(batch_size, num_vertices_idx, k, num_dims).permute(0, 3, 1, 2).contiguous()
    
    return feature