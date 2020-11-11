import torch

from torch import nn
from torch.nn import functional as F


class SAN1d(nn.Module):
    def __init__(self, sparse_activation_list, kernel_size_list):
        super(SAN1d, self).__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1*torch.ones(kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        activations_list = torch.zeros(batch_x.shape[0], len(self.weights_list), *batch_x.shape[1:], device=batch_x.device)
        reconstructions_list = torch.zeros(batch_x.shape[0], len(self.weights_list), *batch_x.shape[1:], device=batch_x.device)
        for index_weights, (sparse_activation, weights) in enumerate(zip(self.sparse_activation_list, self.weights_list)):
            similarity = _conv1d_same_padding(batch_x, weights)
            activations_list[:, index_weights] = sparse_activation(similarity)
            reconstructions_list[:, index_weights] = _conv1d_same_padding(activations_list[:, index_weights], weights)
        reconstruction = reconstructions_list.sum(1)
        return reconstruction


def _conv1d_same_padding(x, weights):
    padding = weights.shape[0] - 1
    odd = int(padding % 2 != 0)
    if odd:
        x = F.pad(x, [0, odd])
    out = F.conv1d(x, weights.unsqueeze(0).unsqueeze(0), padding=padding//2)
    return out


class SAN2d(nn.Module):
    def __init__(self, sparse_activation_list, kernel_size_list):
        super(SAN2d, self).__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1*torch.ones(kernel_size, kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        activations_list = torch.zeros(batch_x.shape[0], len(self.weights_list), *batch_x.shape[1:], device=batch_x.device)
        reconstructions_list = torch.zeros(batch_x.shape[0], len(self.weights_list), *batch_x.shape[1:], device=batch_x.device)
        for index_weights, (sparse_activation, weights) in enumerate(zip(self.sparse_activation_list, self.weights_list)):
            similarity = _conv2d_same_padding(batch_x, weights)
            activations_list[:, index_weights] = sparse_activation(similarity)
            reconstructions_list[:, index_weights] = _conv2d_same_padding(activations_list[:, index_weights], weights)
        reconstruction = reconstructions_list.sum(1)
        return reconstruction


def _conv2d_same_padding(x, weights):
    padding = weights.shape[0] - 1
    odd = int(padding % 2 != 0)
    if odd:
        x = F.pad(x, [0, odd, 0, odd])
    out = F.conv2d(x, weights.unsqueeze(0).unsqueeze(0), padding=padding//2)
    return out
