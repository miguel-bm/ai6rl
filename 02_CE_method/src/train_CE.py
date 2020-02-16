#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import gym
import torch
import numpy as np
from pathlib import Path
from collections import namedtuple
from utils import get_device
from nn_models import DenseNN


def generate_batch(env, model, batch_size: int, discount_factor: float = 1.0):
    """Play a episodes of an environment acording to a model and record results.

    It stores results in named tuples.

    It also can apply a discount factor to the rewards at each step, with
    steps further on in an episode having a higher discount. However, this
    discount has no effet on gameplay here.

    Args:
        env
        model
        batch_size
        discount_factor

    Returns:
        List with all episodes played
    """
    # Define named tuples for recording
    Episode = namedtuple("Episode", field_names=["reward", "steps"])
    Step = namedtuple("Step", field_names=["observation", "action", "reward"])

    # This allows to use model with one input at a time and with numpy arrays
    model_wrapper = lambda obs: torch.nn.Softmax(dim=1)(
                        model(torch.FloatTensor([obs]))).data.numpy()[0]

    # Play and record a batch of episodes with the current model
    batch = []
    for episode in range(batch_size):
        done = False
        reward = 0.0
        episode_steps = []
        obs = env.reset()
        while not done:
            # Get action from sampling on model results
            act_probs = model_wrapper(obs)
            action = np.random.choice(len(act_probs), p=act_probs)

            # Step the model and store results
            next_obs, step_reward, done, _ = env.step(action)
            reward += step_reward
            episode_steps.append(Step(observation=obs,
                                      action=action,
                                      reward=step_reward))
            obs = next_obs
        # Apply the discount to the episode reward (step rewards are unchanged)
        discounted_reward = discount_reward(episode_steps, discount_factor)
        batch.append(Episode(reward=discounted_reward,
                             steps=episode_steps))
    return batch

def discount_reward(steps: list, discount_factor: float=1.0):
    episode_discounted_reward = 0.0
    for i, step in enumerate(steps):
        discounted_reward = step.reward * discount_factor ** i
        episode_discounted_reward += discounted_reward
    return episode_discounted_reward

def filter_episodes(batch: list, elite_num: int, elite_episodes: list,
                    keep_elite: bool=False, min_reward: float=0.0):
    """Given a batch of episodes, keep only a few with the best rewards.

    If keep elite is selected, the best episodes from the previous iterations
    might be kept along with newer ones if the have a better reward.

    A minimum overall reward can also be imposed, which can be useful in order
    to avoid learning from episodes which are bad, if only less bad than others.
    """
    # Erase elite episodes if they are not to be kept
    if not keep_elite:
        keep_elite = []

    # Find the threshold reward for an episode to be kept
    rewards = [episode.reward for episode in batch+elite_episodes]
    reward_bound = np.sort(rewards)[::-1][elite_num-1]

    # Go thorugh the episodes and filter by threshold and minimum reward too
    best_episodes = []
    for episode in batch+elite_episodes:
        if (episode.reward >= max(reward_bound, min_reward)):
            best_episodes.append(episode)

    return best_episodes, reward_bound

def unwrap_episodes(episodes: list):
    """Given a list of episodes, return the observations and actions.

    All observations and actions are flattened across the different episodes,
    and returned as torch tensors, where the first dimension will be of length
    sum([len(episode.steps) for episode in episodes]).
    """
    observations_l = []
    actions_l = []
    for episode in episodes:
        observations_l.extend([step.observation for step in episode.steps])
        actions_l.extend([step.action for step in episode.steps])
    observations = torch.FloatTensor(observations_l)
    actions = torch.LongTensor(actions_l)
    return observations, actions

def train_CE(
    environment: str = typer.Argument("CartPole-v0"),
    elite_ratio: float = typer.Option(
        0.3,
        show_default=True,
        help="Ratio of cases with best score that will be used for training.",
        ),
    reward_objective: float = typer.Option(
        199,
        show_default=True,
        help=("Mean total accumulated reward to reach."),
        ),
    discount_factor: float = typer.Option(
        1.0,
        show_default=True,
        help=("Decrease reward of later steps in an episode.")
        ),
    keep_elite: bool = typer.Option(
        True,
        show_default=True,
        help=("Keep an elite number of episodes across epochs.")
        ),
    min_reward: float = typer.Option(
        0,
        show_default=True,
        help=("Minimum reward for an episode to be in the elite batch.")
        ),
    batch_size: int = typer.Option(
        20,
        show_default=True,
        help="Number of episodes to run per batch.",
        ),
    max_epochs: int = typer.Option(
        100,
        show_default=True,
        help=("Maximum number of epochs of episodes to play on and train."),
        ),
    learning_rate: float = typer.Option(
        0.01,
        show_default=True,
        help=("Proportion of step in weights at each training iteration."),
        ),
    hidden_size: int = typer.Option(
        128,
        show_default=True,
        help="Number of perceptrons in hidden layer of the NN model.",
        ),
    save: bool = typer.Option(
        True,
        show_default=True,
        help="Save the trained model to disk",
        ),
    outdir: Path = typer.Option(
        Path.cwd()/"models",
        show_default=True,
        help=("Output directory for the saving the model "+
              "[default: ./models]."),
        ),
    monitor: bool = typer.Option(
        False,
        show_default=True,
        help="Record videos of the training process.",
        ),
    monitor_outdir: Path = typer.Option(
        Path.cwd()/"mon",
        show_default=True,
        help=("Output directory for the results of the monitor "+
              "[default: ./mon]."),
        ),
    ):
    """Train a Cross-Entropy RL model on a OpenAI Gym environment.

    The model will be a PyTorch neural network with one single hidden layer,
    which will lear a policy mapping observations to actions through the
    Cross-Entropy method.

    Args:
        environment (str): name of the Gym environment to train the model on.

    Returns:
        PyTorch model trained until the reward obejective or the maximum number
        of epochs was reached.
    """

    typer.echo(f"Training a cross-entropy RL model for {environment}")
    if monitor:
        typer.echo(f"Videos of training will be saved to {monitor_outdir}")

    # Set up device for PyTorch
    device, device_name, gpu_available = get_device()

    # Create environment
    env = gym.make(environment)
    if monitor:
        env = gym.wrappers.Monitor(env, directory=monitor_outdir, force=True)
    obs_size = env.observation_space.shape[0]
    actions_size = env.action_space.n

    # Create a model (fully connected NN with 1 hidden layer)
    model = DenseNN(obs_size, [hidden_size], actions_size).to(device)
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    elite_episodes = []
    for epoch in range(max_epochs):
        # Run a batch of cases
        batch = generate_batch(env, model, batch_size, discount_factor)
        reward_mean = np.mean([episode.reward for episode in batch])

        # Filter to elite episodes
        elite_num = int(batch_size * elite_ratio)
        elite_episodes,  reward_bound = filter_episodes(batch, elite_num,
                                                        elite_episodes,
                                                        keep_elite=keep_elite,
                                                        min_reward=min_reward)

        # If there are not enough good episodes yet, do not train on this epoch
        if len(elite_episodes) < elite_num:
            typer.echo("Not enough elite episodes on epoch {epoch}.")
            continue

        # Unwrap data from episodes (all steps on the same level and order)
        observations, actions = unwrap_episodes(elite_episodes)

        # Train the model on the elite episode steps
        optimizer.zero_grad()
        action_scores = model(observations)
        loss = objective(action_scores, actions)
        loss.backward()
        optimizer.step()

        # Report on the epoch stats and update the tensorboard monitor
        typer.echo(f"{epoch:4d} -> Loss: {loss:.3f}  "+
                   f"Mean Reward: {reward_mean:.3f}  "+
                   f"Reward Bound: {reward_bound:.3f}")

        # End the process early if the reward objective was reached
        if reward_mean > reward_objective:
            typer.echo("Latest batch reached the reward objective!")
            break

    typer.echo(f"Process stoped after {epoch} epochs. Saving model...")

    # Save model
    model_name=f"CE_model_fcNN_{environment}.pt"
    torch.save(model, outdir/model_name)
    typer.echo(f"Model saved to {outdir/model_name}")

    return model

if __name__ == "__main__":
    typer.run(train_CE)
