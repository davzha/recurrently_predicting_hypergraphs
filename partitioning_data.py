"""
Adapted from https://github.com/hadarser/SetToGraphPaper/blob/master/dataloaders/jets_loader.py (Apache-2.0 License)
"""

import os
import uproot
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from datetime import datetime

MAX_EDGES = 10

data_dir = 'data/'
node_features_list = ['trk_d0', 'trk_z0', 'trk_phi', 'trk_ctgtheta', 'trk_pt', 'trk_charge']
jet_features_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_M']
true_vtx_list = ['true_vtx_x', 'true_vtx_y', 'true_vtx_z']

def get_data_loader(which_set, batch_size, debug_load=False):
    jets_data = JetGraphDataset(which_set, debug_load)
    batch_sampler = JetsBatchSampler(jets_data.n_nodes, batch_size)
    data_loader = DataLoader(jets_data, batch_sampler=batch_sampler)

    return data_loader


def transform_features(transform_list, arr):
    new_arr = np.zeros_like(arr)
    for col_i, (mean, std) in enumerate(transform_list):
        new_arr[col_i, :] = (arr[col_i, :] - mean) / std
    return new_arr


class JetGraphDataset(Dataset):
    def __init__(self, which_set, debug_load=False, random_permutation=True):
        """
        Initialization
        :param which_set: either "train", "validation" or "test"
        :param debug_load: if True, will load only a small subset
        :param random_permutation: if True, apply random permutation to the order of the nodes/vertices.
        """
        assert which_set in ['train', 'validation', 'test']
        fname = {'train': 'training', 'validation': 'valid', 'test': 'test'}

        self.random_permutation = random_permutation
        self.filename = os.path.join(data_dir, which_set, fname[which_set]+'_data.root')
        with uproot.open(self.filename) as f:
            tree = f['tree']
            self.n_jets = int(tree.numentries)
            self.n_nodes = np.array([len(x) for x in tree.array(b'trk_vtx_index')])

            self.jet_arrays = tree.arrays(
                jet_features_list + node_features_list + true_vtx_list + ['trk_vtx_index'])
            self.sets, self.partitions, self.partitions_as_graphs = [], [], []

        if debug_load:
            self.n_jets = 100
            self.n_nodes = self.n_nodes[:100]

        start_load = datetime.now()

        for set_, partition, partition_as_graph in self.get_all_items():
            if torch.cuda.is_available():
                set_ = torch.tensor(set_, dtype=torch.float, device='cuda')
                partition = torch.tensor(partition, dtype=torch.long, device='cuda')
                partition_as_graph = torch.tensor(partition_as_graph, dtype=torch.float, device='cuda')
            self.sets.append(set_)
            self.partitions.append(partition)
            self.partitions_as_graphs.append(partition_as_graph)

        if not torch.cuda.is_available():
            self.sets = np.array(self.sets)
            self.partitions = np.array(self.partitions)
            self.partitions_as_graphs = np.array(self.partitions_as_graphs)

        print(f' {str(datetime.now() - start_load).split(".")[0]}', flush=True)

    def __len__(self):
        """Returns the length of the dataset"""
        return self.n_jets

    def get_all_items(self):
        node_feats = np.array([np.asarray(self.jet_arrays[str.encode(x)]) for x in node_features_list])
        jet_feats = np.array([np.asarray(self.jet_arrays[str.encode(x)]) for x in jet_features_list])
        vtx_feats = np.array([np.asarray(self.jet_arrays[str.encode(x)]) for x in true_vtx_list])
        n_labels = np.array(self.jet_arrays[b'trk_vtx_index'])

        for i in range(self.n_jets):
            n_nodes = self.n_nodes[i]
            node_feats_i = np.stack(node_feats[:, i], axis=0)  # shape (6, n_nodes)
            jet_feats_i = jet_feats[:, i]  # shape (4, )
            jet_feats_i = jet_feats_i[:, np.newaxis]  # shape (4, 1)
            vtx_feats_i = np.stack(vtx_feats[:, i], axis=0)

            node_feats_i = transform_features(FeatureTransform.node_feature_transform_list, node_feats_i)
            jet_feats_i = transform_features(FeatureTransform.jet_features_transform_list, jet_feats_i)
            vtx_feats_i = transform_features(FeatureTransform.true_vtx_transform_list, vtx_feats_i)

            jet_feats_i = np.repeat(jet_feats_i, n_nodes, axis=1)  # change shape to (4, n_nodes)
            set_i = np.concatenate([node_feats_i, jet_feats_i]).T  # shape (n_nodes, 10)

            partition_i = n_labels[i]

            incidence_i = (np.arange(MAX_EDGES)[:,np.newaxis] == n_labels[i]).astype("float")
            vtx_feats_i = np.concatenate((
                vtx_feats_i.T, np.zeros((MAX_EDGES - vtx_feats_i.shape[1], 3))
                ))

            if self.random_permutation:
                perm = np.random.permutation(n_nodes)
                set_i = set_i[perm]  # random permutation
                partition_i = partition_i[perm]  # random permuatation
                incidence_i = incidence_i[:,perm]

            target_i = incidence_i

            yield set_i, partition_i, target_i

    def __getitem__(self, idx):
        """Generates a single instance of data"""
        return self.sets[idx], self.partitions[idx], self.partitions_as_graphs[idx]


class JetsBatchSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the jets
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]


class FeatureTransform(object):
    # Based on mean and std values of TRAINING set only
    node_feature_transform_list = [
        (0.0006078152, 14.128961),
        (0.0038490593, 10.688491),
        (-0.0026713554, 1.8167108),
        (0.0047640945, 1.889725),
        (5.237357, 7.4841413),
        (-0.00015662189, 1.0)]

    jet_features_transform_list = [
        (75.95093, 49.134453),
        (0.0022607117, 1.2152709),
        (-0.0023569583, 1.8164033),
        (9.437994, 6.765137)]

    true_vtx_transform_list = [
        (0.15225221, 91.49316),
        (-0.013348891, 91.526245),
        (0.09531866, 221.47397)
    ]