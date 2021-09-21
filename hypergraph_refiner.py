import torch
from torch import nn


class IterativeRefiner(nn.Module):
    def __init__(self, n_edges, d_in, d_hid, T):
        super().__init__()
        self.n_edges = n_edges
        self.T = T
        self.d_in = d_in
        self.d_hid = d_hid

        self.proj_inputs = nn.Linear(d_in, d_hid)
        self.refiner = HypergraphRefiner(d_hid)

        self.edges_mu = nn.Parameter(torch.randn(1, 1, d_hid))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, 1, d_hid))
        nn.init.xavier_uniform_(self.edges_logsigma)

    def get_initial(self, inputs, n_edges=None):
        b, n_v, _, device = *inputs.shape, inputs.device
        n_e = n_edges if n_edges is not None else self.n_edges

        mu = self.edges_mu.expand(b, n_e, -1)
        sigma = self.edges_logsigma.exp().expand(b, n_e, -1)
        e_t = mu + sigma * torch.randn(mu.shape, device = device)

        v_t = self.proj_inputs(inputs)

        i_t = torch.zeros(b, n_e, n_v, device=device)
        return e_t, v_t, i_t

    def forward(self, inputs, e_t, v_t, i_t, t_skip=None, t_bp=None):
        t_skip = 0 if t_skip is None else t_skip
        t_bp = self.T if t_bp is None else t_bp
        inputs = self.proj_inputs(inputs)
        pred_bp = []

        with torch.no_grad():
            for _ in range(t_skip):
                p, e_t, v_t, i_t = self.refiner(inputs, e_t, v_t, i_t)

        for _ in range(t_skip, t_skip+t_bp):
            p, e_t, v_t, i_t = self.refiner(inputs, e_t, v_t, i_t)
            pred_bp.append(p)

        return pred_bp, e_t, v_t, i_t


class StackedRefiner(IterativeRefiner):
    def __init__(self, n_edges, d_in, d_hid, T):
        super().__init__(n_edges, d_in, d_hid, T)
        self.refiner = nn.ModuleList([HypergraphRefiner(d_hid) for _ in range(T)])
        
    def forward(self, inputs, e_t, v_t, i_t, t_skip=None, t_bp=None):
        t_skip = 0 if t_skip is None else t_skip
        t_bp = self.T if t_bp is None else t_bp
        inputs = self.proj_inputs(inputs)
        pred_post = []

        with torch.no_grad():
            for i in range(t_skip):
                p, e_t, v_t, i_t = self.refiner[i](inputs, e_t, v_t, i_t)

        for i in range(t_skip, t_skip+t_bp):
            p, e_t, v_t, i_t = self.refiner[i](inputs, e_t, v_t, i_t)
            pred_post.append(p)

        return pred_post, e_t, v_t, i_t


class HypergraphRefiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_e = DeepSet(2*dim, [dim, dim])
        self.mlp_n = DeepSet(3*dim, [dim, dim])

        self.norm_pre_n  = nn.LayerNorm(3*dim)
        self.norm_pre_e  = nn.LayerNorm(2*dim)
        self.norm_n = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)

        self.mlp_incidence = nn.Sequential(
            OutCatLinear(dim, dim, 1, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.edge_indicator = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, e_t, n_t, i_t):
        i_t = self.mlp_incidence((e_t, n_t, i_t)).squeeze(3)
        e_ind = self.edge_indicator(e_t)
        im_t = i_t * e_ind

        updates_e = torch.einsum("ben,bnd->bed", im_t, n_t)
        e_t = self.norm_e(e_t + self.mlp_e(self.norm_pre_e(torch.cat([e_t, updates_e], dim=-1))))

        updates_n = torch.einsum("ben,bed->bnd", im_t, e_t)
        n_t = self.norm_n(n_t + self.mlp_n(self.norm_pre_n(torch.cat([inputs, n_t, updates_n], dim=-1))))
    
        pred = torch.cat([i_t, e_ind], dim=2)
        return pred, e_t, n_t, i_t


class OutCatLinear(nn.Module):
    def __init__(self, d_e, d_n, d_i, d_out):
        super().__init__()
        self.proj_e = nn.Linear(d_e, d_out)
        self.proj_n = nn.Linear(d_n, d_out)
        self.proj_i = nn.Linear(d_i, d_out)

    def forward(self, inputs):
        e_t, n_t, i_t = inputs
        o0 = self.proj_n(n_t).unsqueeze(1)
        o1 = self.proj_e(e_t).unsqueeze(2)
        o2 = self.proj_i(i_t.unsqueeze(3))
        return o0 + o1 + o2


class DeepSet(nn.Module):
    def __init__(self, d_in, d_hids):
        super().__init__()
        layers = []
        layers.append(DeepSetLayer(d_in, d_hids[0]))
        for i in range(1, len(d_hids)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(d_hids[i-1]))
            layers.append(DeepSetLayer(d_hids[i-1], d_hids[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x0 = self.layer1(x)
        x1 = self.layer2(x - x.mean(dim=1, keepdim=True))
        x = x0 + x1
        return x