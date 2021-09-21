from numpy.random import sample, shuffle
import torch
import scipy.spatial
import random

def sample_random(n, d):
    points = torch.rand(n, d)
    c = scipy.spatial.Delaunay(points)
    facets = torch.tensor(c.simplices)
    a = torch.arange(n)
    inc = (facets.unsqueeze(2) == a.view(1,1,n)).sum(1)

    adj = inc.T @ inc
    adj = (adj > 0).int()
    adj[torch.eye(n).bool()] = 0
    return points, adj.float()

class DelaunayTriangulationData(torch.utils.data.Dataset):
    def __init__(self, n_range, dim, length) -> None:
        super().__init__()
        self.n_range = n_range
        self.dim = dim
        self.length = length

        self.max_facets = 0
        self.points = []
        self.n_points = []
        self.incidence = []
        self.fill_samples()

    def fill_samples(self):
        for _ in range(self.length):
            n = self.n_range[torch.randint(0,len(self.n_range), ())]
            V, I = sample_random(n, self.dim)
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

def get_dt_dl(m, batch_size, n_range, dim, max_facets = None, length=None, bucket=True):
    if m == "train":
        length =  50000 if length is None else length
        shuffle = True

    else:
        length = 5000 if length is None else length
        shuffle = False

    dataset = DelaunayTriangulationData(n_range=n_range,dim=dim,length=length)
    max_facets = dataset.max_facets if max_facets is None else max_facets
    if batch_size > 1 and bucket:
        sampler = BucketSampler(dataset, batch_size, dataset.n_points, shuffle=shuffle)
        batch_size = 1
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler,
        batch_size=batch_size,
        num_workers=10)