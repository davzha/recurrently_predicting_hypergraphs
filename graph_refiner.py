import torch
import torch.nn as nn
import torch.nn.functional as F

from hypergraph_refiner import DeepSet


class MLPAdjacency(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_x = nn.Linear(dim, dim)
        self.proj_i = nn.Linear(1, dim)
        self.proj_s = nn.Linear(dim, 1)

    def forward(self, x, inc):
        x = self.proj_x(x)
        inc = self.proj_i(inc.unsqueeze(3))
        s = F.relu(x.unsqueeze(1) + x.unsqueeze(2) + inc)
        return torch.sigmoid(self.proj_s(s))


class GraphRefiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_n = DeepSet(3*dim, [dim, dim])

        self.norm_pre_n  = nn.LayerNorm(3*dim)
        self.norm_n = nn.LayerNorm(dim)

        self.mlp_incidence = MLPAdjacency(dim)

    def forward(self, inputs, n_t, i_t):
        i_t = self.mlp_incidence(n_t, i_t).squeeze(3)

        updates_n = torch.einsum("ben,bed->bnd", i_t, n_t)
        n_t = self.norm_n(n_t + self.mlp_n(self.norm_pre_n(torch.cat([inputs, n_t, updates_n], dim=-1))))
        
        pred = i_t
        return pred, n_t, i_t


class IterativeRefiner(nn.Module):
    def __init__(self, d_in, d_hid, T):
        super().__init__()
        self.T = T
        self.d_in = d_in
        self.d_hid = d_hid

        self.proj_inputs = nn.Linear(d_in, d_hid)
        self.refiner = GraphRefiner(d_hid)

    def get_initial(self, inputs):
        b, n_v, _, device = *inputs.shape, inputs.device

        v_t = self.proj_inputs(inputs)

        i_t = torch.zeros(b, n_v, n_v, device=device)
        return v_t, i_t

    def forward(self, inputs, v_t, i_t, t_skip=None, t_bp=None):
        t_skip = 0 if t_skip is None else t_skip
        t_bp = self.T if t_bp is None else t_bp
        inputs = self.proj_inputs(inputs)
        pred_bp = []

        with torch.no_grad():
            for _ in range(t_skip):
                p, v_t, i_t = self.refiner(inputs, v_t, i_t)

        for _ in range(t_skip, t_skip+t_bp):
            p, v_t, i_t = self.refiner(inputs, v_t, i_t)
            pred_bp.append(p)

        return pred_bp, v_t, i_t
