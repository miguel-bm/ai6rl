#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNN(nn.Module):

    def __init__(self, insize, outsize, densesize, densenum):
        super(DenseNN, self).__init__()
        self.fc_in = nn.Linear(insize, densesize)
        self.fcs = [nn.Linear(densesize, densesize) for i in range(densenum)]
        self.fc_out = nn.Linear(densesize, outsize)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for layer in self.fcs:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return x
