"""Slot attention code adapted from https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py (MIT License)
"""

import torch
from torch import nn
from torch.nn import init

from hypergraph_refiner import DeepSet


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        # self.mlp = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.ReLU(inplace = True),
        #     nn.Linear(hidden_dim, dim)
        # )
        self.mlp = DeepSet(dim, [hidden_dim, dim])

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)
            attn_ = attn + self.eps
            attn_ = attn_ / attn_.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn_)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn


class SASet2Hypergraph(nn.Module):
    def __init__(self, max_k, d_in, d_hid, T):
        super().__init__()
        self.enc = DeepSet(d_in, [d_hid, d_hid])
        self.set2set = SlotAttention(max_k, d_hid, hidden_dim=d_hid, iters=T)
        self.mlp_out = nn.Sequential(
            nn.Linear(2 * d_hid, d_hid),
            nn.ReLU(inplace=True),
            nn.Linear(d_hid, 1),
            nn.Sigmoid()
        )
        self.edge_ind = nn.Sequential(
            nn.Linear(d_hid, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc(x)
        e, _ = self.set2set(x)
        ind = self.edge_ind(e)

        n_nodes = x.size(1)
        n_edges = e.size(1)
        outer = torch.cat([
            x.unsqueeze(1).expand(-1, n_edges, -1, -1), 
            e.unsqueeze(2).expand(-1, -1, n_nodes, -1)], dim=3)
        incidence = self.mlp_out(outer).squeeze(3)

        return torch.cat([incidence, ind], dim=-1)