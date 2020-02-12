#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNN(nn.Module):

    def __init__(self, insize, outsize, densesize):
        super(DenseNN, self).__init__()
        self.fc_in = nn.Linear(insize, densesize)
        self.fc_h1 = nn.Linear(densesize, densesize)
        self.fc_out = nn.Linear(densesize, outsize)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_h1(x))
        x = self.fc_out(x)
        return x
