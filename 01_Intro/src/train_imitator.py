#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from utils import *
from nn_models import DenseNN
import torch.nn.functional as F


def train_imitator(
        gameplay: Path,
        name: str,
        epochs: int,
        test_ratio: float = typer.Option(
            0.2,
            help=("Ratio of entries into test set"),
            ),
        val_ratio: float = typer.Option(
            0.125,
            help=("Ratio of entries into validation set"),
            ),
        batch_size: int = typer.Option(
            50,
            help=("Size of minibatches within an epoch for training"),
            ),
        dense_size: int = typer.Option(
            512,
            help=("Size of dense hidden layers."),
            ),
        dense_num: int = typer.Option(
            2,
            help=("Number of dense hidden layers."),
            ),
        outdir: Path = typer.Option(
            Path.cwd()/"models",
            help=("Output directory for the trained model"+
                  "[default: ./models]"),
            ),
        ):

    # Set up device
    device, device_name, gpu_available = get_device()

    # Load dataset
    gameplay_df = pd.read_pickle(gameplay)
    observations = gameplay_df["observation"]
    actions = gameplay_df["action"]

    # Transform actions into a space in 0..N-1
    action_space = list(actions.value_counts().index)
    action_space_len = len(action_space)
    action_encoding = {x: i for i, x in enumerate(action_space)}
    action_decoding = {i: x for i, x in enumerate(action_space)}
    enconded_actions = actions.apply(lambda x: action_encoding[x])

    # Interpret as arrays
    X = np.array(list(observations.apply(
        lambda x: np.array(x.flatten()))))  # Flatten observations if multidim
    y = np.array(list(enconded_actions))

    # Store as torch tensors
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    train_dl, val_dl, test_dl = get_dataloaders(X, y,
                                                test_ratio, val_ratio,
                                                batch_size)

    # Set up model
    learning_rate = 0.05
    momentum      = 0.1

    loss_func = F.cross_entropy
    model = DenseNN(X.shape[1], action_space_len,
                    dense_size, dense_num)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)

    # Train model
    epoch_record, train_loss_record, test_loss_record, test_acc_record = fit(
        epochs, model, loss_func, optimizer, train_dl, val_dl)

    # Test model
    test_loss, bs = zip(
        *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl])
    test_loss = np.average(test_loss, weights=bs)

    test_acc, bs = zip(
        *[acc_batch(model, xb, yb) for xb, yb in test_dl])
    test_acc = np.average(test_acc, weights=bs)

    typer.echo(f"Test loss: {test_loss}    Test accuracy: {test_acc}")

    # Save the model
    torch.save(model.state_dict(), outdir/name)


if __name__ == "__main__":
    typer.run(train_imitator)
