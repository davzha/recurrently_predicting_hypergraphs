import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EPS = 1e-8

def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, s, e):
    return np.array([linear_sum_assignment(p) for p in arr[s:e]])

def ray_lsa(arr, n):
    l = arr.shape[0]
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    res = [lsa.remote(arr_id, ind[i], ind[i+1]) for i in range(n)]
    res = np.concatenate([ray.get(r) for r in res])
    return res

def LAP_loss(input, target, n=0):
    pdist = F.binary_cross_entropy(
        input.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, input.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    if n > 0:
        indices = ray_lsa(pdist_, n)
    else:
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
    total_loss = losses.mean(1)

    return total_loss

def _error_count_indicator(gt_inc, pred_inc, d):
    pred_m = pred_inc[...,-1] > 0.5
    gt_m = gt_inc[...,-1] > 0.5

    pred_inc = pred_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]
    gt_inc = gt_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]

    # batch x edge_pred x edge_gt
    eq = (pred_inc.unsqueeze(2) == gt_inc.unsqueeze(1)).all(3)
    eq = eq * pred_m.unsqueeze(2) * gt_m.unsqueeze(1)
    tp = eq.any(1).sum(1)  # count unique only
    fp = (pred_m * ~eq.any(2)).sum(1)
    fn = (gt_m * ~eq.any(1)).sum(1)
    return tp, fp, fn

def _triu_mean(x):
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    return x.triu(1).sum((1,2)) * 2. / (x.size(1) * (x.size(1)-1))
    
def _error_count_adj(gt_adj, pred_adj):
    pred_adj = pred_adj.clamp(0, 1)
    tp = _triu_mean(gt_adj * pred_adj)
    fp = _triu_mean((1 - gt_adj) * pred_adj)
    fn = _triu_mean(gt_adj * (1 - pred_adj))
    return tp, fp, fn

def error_count(type, gt, pred, **kwargs):
    assert type in ["adj", "ind"]
    if type == "adj":
        tp, fp, fn = _error_count_adj(gt, pred)
    else:
        tp, fp, fn = _error_count_indicator(gt, pred, kwargs.get("d_feats"))
    return tp, fp, fn

def precision(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fp + EPS)

def recall(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fn + EPS)

def f1_score(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
    return f1

def delaunay_adj_metrics(targ_adj, pred_adj, k=2):
    diag_mask = torch.eye(pred_adj.shape[2]).repeat(pred_adj.shape[0], 1, 1).bool()
    pred_adj = (pred_adj > 0.5).int()
    pred_adj[diag_mask] = 0

    tp = (targ_adj * pred_adj).sum((1,2)).float()
    tn = ((1-targ_adj) * (1-pred_adj)).sum((1,2)).float()
    fp = ((1-targ_adj) * pred_adj).sum((1,2)).float()
    fn = (targ_adj * (1-pred_adj)).sum((1,2)).float()
    
    acc = ((tp+tn) / (tp+tn+fp+fn))
    prec = (tp / (tp+fp+EPS))
    rec = (tp / (tp+fn+EPS))
    fone = 2*tp / (2*tp+fp+fn+EPS)
    return acc, fone, prec, rec

def mae_cardinality(pred, target):
    card_targ = (pred[:,:,-1]>0.5).sum(1).float()
    card_pred = (target[:,:,-1]>0.5).sum(1).float()
    return F.l1_loss(card_targ, card_pred)


