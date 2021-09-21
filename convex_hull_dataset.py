from numpy.random import sample, shuffle
import torch
import torch.nn.functional as F
import scipy.spatial
import random

def sample_random(n, d, unit=False):
    points = torch.randn(n, d)
    if unit:
        points = points / points.norm(dim=1, keepdim=True)
    c = scipy.spatial.ConvexHull(points)
    facets = torch.tensor(c.simplices)
    a = torch.arange(n)
    inc = (facets.unsqueeze(2) == a.view(1,1,n)).sum(1)
    return points, inc

class ConvexHullData(torch.utils.data.Dataset):
    def __init__(self, n_range, dim, unit_norm, length) -> None:
        super().__init__()
        self.n_range = n_range
        self.dim = dim
        self.unit_norm = unit_norm
        self.length = length

        self.max_facets = 0
        self.points = []
        self.n_points = []
        self.incidence = []
        self.fill_samples()

    def fill_samples(self):
        for _ in range(self.length):
            n = self.n_range[torch.randint(0,len(self.n_range), ())]
            V, I = sample_random(n, self.dim, self.unit_norm)
            if I.size(0) > self.max_facets:
                self.max_facets = I.size(0)
            self.points.append(V)
            self.incidence.append(I)
            self.n_points.append(V.size(0))

    def __getitem__(self, index):
        return self.points[index], self.incidence[index]

    def __len__(self):
        return len(self.points)


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, set_sizes, shuffle=False) -> None:
        super().__init__(data_source)
        self.batch_size = batch_size
        self.set_sizes = torch.tensor(set_sizes)
        self.shuffle = shuffle
        self.len = len(self.get_batch_idx())
    
    def get_batch_idx(self):
        uniq_sizes = self.set_sizes.unique()
        batches = []
        for s in uniq_sizes:
            idx = torch.nonzero(self.set_sizes == s).squeeze()
            if self.shuffle:
                idx = idx[torch.randperm(idx.size(0))]
            b = torch.split(idx, self.batch_size)
            batches.extend(b)
        if self.shuffle:
            random.shuffle(batches)
        return batches
            
    def __iter__(self):
        batches = self.get_batch_idx()
        for b in batches:
            yield b

    def __len__(self):
        return self.len


def get_collate_fn(max_facets, add_indicator=False):
    if not add_indicator:
        def collate_fn(batch):
            points = []
            incidence = []
            for p, i in batch:
                points.append(p)
                incidence.append(torch.cat([i, torch.zeros(max_facets - i.size(0), i.size(1))],dim=0))
            return torch.stack(points), torch.stack(incidence)
        return collate_fn
    else:
        def collate_fn(batch):
            points = []
            incidence = []
            for p, i in batch:
                nf = i.size(0)
                inc = torch.cat([i, torch.zeros(max_facets - nf, i.size(1))],dim=0)
                inc = torch.cat([inc, torch.zeros(max_facets, 1)], dim=1)
                inc[:nf,-1] = 1.
                incidence.append(inc)
                points.append(p)
            return torch.stack(points), torch.stack(incidence)
        return collate_fn

def get_ch_dl(m, batch_size, n_range, dim, max_facets = None, unit_norm = True, add_indicator=False, length=None, bucket=True):
    if m == "train":
        length =  20000 if length is None else length
        shuffle = True

    else:
        length = 2000 if length is None else length
        shuffle = False

    dataset = ConvexHullData(n_range=n_range,dim=dim,unit_norm=unit_norm,length=length)
    max_facets = dataset.max_facets if max_facets is None else max_facets
    collate_fn = get_collate_fn(max_facets, add_indicator=add_indicator)
    if batch_size > 1 and bucket:
        sampler = BucketSampler(dataset, batch_size, dataset.n_points, shuffle=shuffle)
        batch_size = 1
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=10)