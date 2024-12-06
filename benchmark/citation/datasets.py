import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name == "Reddit":
        dataset = Reddit(path)
    elif name == "Arxiv":
        dataset = PygNodePropPredDataset('ogbn-arxiv',
                                         osp.join(osp.dirname(path), "OGB"))
    else:
        dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset
