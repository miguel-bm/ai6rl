#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNN_1h(nn.Module):
    def __init__(self, obs_size, hidden_size, actions_size):
        super(DenseNN, self).__init__()
        self.fc_in  = nn.Linear(obs_size,    hidden_size)
        self.fc_out = nn.Linear(hidden_size, actions_size)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.fc_out(x)
        return x


class DenseNN(nn.Module):
    """Generic implementation of feedforward dense Neural Netwrok.

    It includes only ReLU activations and no regularization.

    Attributes:
        in_size (int): size of input
        hidden_sizes (list): sizes of hidden layers
        out_size (int): size of output
    """
    def __init__(self, in_size, hidden_sizes, out_size):
        super(DenseNN, self).__init__()
        from collections import OrderedDict
        # Layers is a list of ("layer_name", layer) tuples, includes activations
        layers = [("linear_1", nn.Linear(in_size, hidden_sizes[0])),
                  ("relu_1", nn.ReLU())]
        for i in range(len(hidden_sizes)-1):
            layers.append((f"linear_{i+2}", nn.Linear(hidden_sizes[i],
                                                      hidden_sizes[i+1])))
            layers.append((f"relu_{i+2}", nn.ReLU()))
        i = len(hidden_sizes)-2
        layers.append((f"linear_{i+3}", nn.Linear(hidden_sizes[-1],
                                                  out_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)
