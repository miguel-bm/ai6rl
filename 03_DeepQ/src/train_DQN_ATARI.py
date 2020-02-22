#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models import wrappers
from models import dqn_model

import time
import numpy as np
import collections
import typer
from pathlib import Path

import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.simplefilter("error")
warnings.simplefilter("ignore", UserWarning)


Experience = collections.namedtuple('Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, discount_factor, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = (next_state_values * discount_factor +
                                    rewards_v)
    return nn.MSELoss()(state_action_values, expected_state_action_values)



def train_DQL_ATARI(
    environment: str = typer.Argument("PongNoFrameskip-v4"),
    reward_bound: float = typer.Option(
        18.0,
        show_default=True,
        help = "Reward bjective to reach for the loop to stop.",
        ),
    discount_factor: float = typer.Option(
        0.99,
        show_default=True,
        help = "Decrease reward of later steps in a trajectory.",
        ),
    batch_size: int = typer.Option(
        32,
        show_default=True,
        help = "Number of state samples for the backward pass.",
        ),
    replay_size: int = typer.Option(
        10000,
        show_default=True,
        help = "Number of state samples for the backward pass.",
        ),
    learning_rate: float = typer.Option(
        1e-4,
        show_default=True,
        help = "Rate of step in weights at every backward pass.",
        ),
    sync_target_frames: int = typer.Option(
        1000,
        show_default=True,
        help = "Number of frames before target net is updated with best.",
        ),
    replay_start_size: int = typer.Option(
        10000,
        show_default=True,
        help = "Frames at the beginning before training starts.",
        ),
    epsilon_decay_last_frame: int = typer.Option(
        100000,
        show_default=True,
        help = "Frames until epsilon stops decaying.",
        ),
    epsilon_start: float = typer.Option(
        1.0,
        show_default=True,
        help = "Probability of random step at beginning of training.",
        ),
    epsilon_final: float = typer.Option(
        0.02,
        show_default=True,
        help = "Probability of random step after end of decay.",
        ),
    cuda: bool = typer.Option(
        True,
        show_default=True,
        help = "Use GPU with CUDA if available.",
        ),
    save: bool = typer.Option(
        True,
        show_default=True,
        help = "Save the trained model to disk.",
        ),
    save_name: str = typer.Option(
        None,
        show_default=False,
        help = "Name for the saved model."
        ),
    outdir: Path = typer.Option(
        Path.cwd()/"models",
        show_default=True,
        help= ("Output directory for the saving the model " +
                "[default: ./models]."),
        ),
    ):
    """Train a Deep Q-learning model for playing an ATARI game.


    Args:
        environment (str): name of the Gym environment to train the model on.
    """

    typer.echo(f"Training a Deep Q-learning RL model for {environment}")

    # Set up device for training the network (not inference)
    device = torch.device("cuda" if cuda else "cpu")

    # Make the environment with the proper wrappers
    env = wrappers.make_env(environment)

    # Create the CNN models that will approximate the Q function
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n
        ).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n
        ).to(device)

    # Set up buffer of previous experiences for sampling during training
    buffer = ExperienceBuffer(replay_size)
    agent = Agent(env, buffer)

    epsilon_f = lambda i: max(epsilon_final,
                              epsilon_start - (epsilon_start - epsilon_final) *
                              i / epsilon_decay_last_frame)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    # Training loop
    while True:
        frame_idx += 1
        epsilon = epsilon_f(frame_idx)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            typer.echo(f"{frame_idx:d}: done {len(total_rewards):d} games, " +
                       f"mean reward {mean_reward:.3f}, eps {epsilon:.2f}, " +
                       f"speed {speed:.2f} f/s")

            if best_mean_reward is None or best_mean_reward < mean_reward:
                if save_name is None:
                    save_name = environment + "-best.pt"
                torch.save(net.state_dict(), outdir/save_name)
                if best_mean_reward is not None:
                    typer.echo(f"Best mean reward updated " +
                               f"{best_mean_reward:.3f} -> " +
                               f"{mean_reward:.3f}, model saved")
                best_mean_reward = mean_reward

            if mean_reward > reward_bound:
                typer.echo(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < replay_start_size:
            continue

        if frame_idx % sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, net, tgt_net, discount_factor, device=device)
        loss_t.backward()
        optimizer.step()


if __name__ == "__main__":
    typer.run(train_DQL_ATARI)
