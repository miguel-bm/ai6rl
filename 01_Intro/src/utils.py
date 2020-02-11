#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

def split_off_testset(dataset, test_ratio, seed=0):
    """Split-off a portion of a dataset in order to use it as a test set.

    It uses a fixed seed in order to be repeatable. In principle, the test set
    set should not be used again until the final evaluation, and the remaining
    data should be split into training data and validation data, or used for
    cross-validation.

    TODO:
        - extendable splitting when new data is available
        - stratified splitting
        - ratio check
    """
    test_len = int(len(dataset) * test_ratio)
    lengths = [len(dataset) - test_len, test_len]

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    indices = list(np.random.permutation(sum(lengths)))

    new_ds, testset = [torch.utils.data.dataset.Subset(
        dataset, indices[offset - length:offset])
        for offset, length in zip(torch._utils._accumulate(lengths), lengths)]

    return new_ds, testset

def train_val_split(dataset, val_ratio, seed=None):
    """Split a dataset into training and validation.

    It uses a non-fixed seed by default, but it can be activated for
    repeatibility.
    """
    val_len = int(len(dataset) * val_ratio)
    lengths = [len(dataset) - val_len, val_len]

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    indices = list(np.random.permutation(sum(lengths)))

    train_set, val_set = [torch.utils.data.dataset.Subset(
        dataset, indices[offset - length:offset])
        for offset, length in zip(torch._utils._accumulate(lengths), lengths)]

    return train_set, val_set

def get_dataloaders(X, y, test_ratio, val_ratio, batch_size=50):
    """From data and label tensors, get the DataLoader objects for training and
    evaluation.

    TODO:
        - ratio cehcks
    """
    val_ratio =  val_ratio / (1.-test_ratio)
    dataset = torch.utils.data.TensorDataset(X, y)

    dataset, testset = split_off_testset(dataset, test_ratio)
    trainset, valset = train_val_split(dataset, val_ratio)

    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl

def get_device(try_gpu=True, verbose=True):
    """Get a device to run your model on.

    It will try to get a GPU using CUDA by default. Otherwise, it wll get a CPU.
    """
    gpu_available = torch.cuda.is_available()
    device_name = "cuda" if (gpu_available and try_gpu) else "cpu"
    device = torch.device(device_name)
    if verbose: print(f"Using {device_name}")
    return device, device_name, gpu_available

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    # If an optimizer is used, then backprop and step the model
    if opt is not None:
        loss.backward()  # This method automatically uses the backward graph
                         # calculated during the forward pass to calculate .grad
        opt.step()  # Automatically updates model parameters based on optimizer
                    # rule and current gradient stored on .grad parameter
        opt.zero_grad()  # Reset .grad, otherwise it acumulates

    # Since loss is 0-dimensional, .item() will return it as standard Python num
    return loss.item(), len(xb)

def acc_batch(model, xb, yb):
    max_vals, max_indices = torch.max(model(xb), 1)  # Maximum along output dim
                                                     # (0 is batch dimension)
    # See where the assigned output matches with the labels
    accuracy = float((max_indices == yb).sum().float()/len(yb))
    return accuracy, len(yb)

def fit(epochs, model, loss_func, opt, train_dl, val_dl, verbose=True):
    """Train a torch model.

    It fits torch model to a set of training data acording to an optimizer in
    order to minimize a certain loss function.

    At each epoch, it evaluates the model over a set of validation data and
    records the performance.

    Attributes:
        epochs (int): number of times the model will be exposed to all of the
            training data.
        model: normally a torch model, model(X) -> y, with .train() and .eval()
            methods to toggle it into training or evaluation modes.
        loss_func: a function, such as torch.nn.functional.cross_entropy, which
            takes two torch tensors, and returns the measurement of loss
            between them as another torch tensor, loss(y', y) -> l.
        opt: optimizer criterion for training the model, from torch.optim or
            derived. Must have .step() and .zero_grad() methods.
        train_dl: data loader that iterates over bacthes of the training data.
        val_dl: data loader that iterates over batches of the validation data.
        verbise (bool, Optional): if True, prints out the performance at each
            epoch.

    Returns:
        list: numbers of the epochs the model went through
        list: losses over the training data at every epoch
        list: losses over the validation data at every epoch
        list: accuracy over the validation data at every epoch
    """

    # Initialize the lists were progress will be recorded
    epoch_record, train_loss_record, val_loss_record, val_acc_record = \
        [], [], [], []


    # For each epoch, train the model over the whole dataset in batches
    for epoch in range(epochs):

        # Training
        model.train()  # Sets the model to training mode (backprop after eval)
        train_losses, batch_lens = zip(
            *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
            )  # Evaluates the cross entropy per batch, backpropagates and steps
        train_loss = np.sum(np.multiply(train_losses, batch_lens))/np.sum(batch_lens)
        #train_loss = np.average(train_losses, weights=batch_lens)

        # Evaluation
        model.eval()  # Activate evaluation mode (dropout, batchnorm)
        with torch.no_grad():  # Deactivates autograd engine, saves memory
            val_losses, batch_lens = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in val_dl]
                )  # Evaluates the cross entropy per val. batch, no backprop
            val_accs, _ = zip(
                *[acc_batch(model, xb, yb) for xb, yb in val_dl]
                )  # Calculate accuracy as well

        val_loss = np.sum(np.multiply(val_losses, batch_lens))/np.sum(batch_lens)
        #val_loss = np.average(val_losses, weights=batch_lens)
        val_acc = np.sum(np.multiply(val_accs, batch_lens)) /np.sum(batch_lens)
        #val_acc = np.average(val_accs, weights=batch_lens)

        # Record results for this epoch
        epoch_record.append(epoch)
        train_loss_record.append(train_loss)
        val_loss_record.append(val_loss)
        val_acc_record.append(val_acc)

        # Print results for this epoch
        if verbose:
            print(f"Epoch: {(epoch+1):3}    Train loss: {train_loss:7.5f}    "+
                  f"Val loss: {val_loss:7.5f}    Val accuracy: {val_acc: 5.3f}")

    return epoch_record, train_loss_record, val_loss_record, val_acc_record
