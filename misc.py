import numpy as np

def partitionfunc(n,k,l=1):
    '''
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    Adapted from https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
    '''
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

class IntegerPartitionSampler:
    def __init__(self, n, k, rng):
        self.partitions = np.array(list(partitionfunc(n, k, 0)))
        self.rng = rng

    def __call__(self):
        return self.rng.permutation(self.rng.choice(self.partitions))