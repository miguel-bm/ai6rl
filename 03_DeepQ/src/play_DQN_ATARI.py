#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import time
import gym
import torch
import collections
import numpy as np
from pathlib import Path

import pygame
from pygame.locals import VIDEORESIZE

import gym.wrappers as gymwr

from models import wrappers
from models import dqn_model


class ModelAgent(object):
    """Acts acording to loaded Pytorch model."""
    def __init__(self, model_path):
        # Load the model and set it to inference.
        self.model = torch.load(model_path)
        self.model.eval()

    def act(self, observation, reward, done):
        obs = torch.FloatTensor([observation])
        action_probs = torch.nn.Softmax(dim=1)(self.model(obs)).data.numpy()[0]
        action = np.argmax(action_probs)
        return action


def play_gym_model(
        game: str = typer.Argument("PongNoFrameskip-v4"),
        model: Path = typer.Argument("models/DQL-PongNoFrameskip-v4-mr18.pt"),
        episodes: int = typer.Option(
            1,
            show_default=True,
            help="Number of runs of the environment to simulate.",
            ),
        fps: int = typer.Option(
            30,
            show_default=True,
            help="Frames per second (approximately)."
            ),
        zoom: float = typer.Option(
            1.0,
            show_default=True,
            help="Zoom for the game display."
            ),
        verbose: bool = typer.Option(
            False,
            show_default=True,
            help="Print action, reward and observation at every step.",
            ),
        embeddings: bool = typer.Option(
            False,
            show_default=True,
            help="Record last layer embeddings for a tSNE representation."
            ),
        monitor: bool = typer.Option(
            False,
            show_default=True,
            help="Activate a monitor to record a video of the results."
            ),
        logger: str = typer.Option(
            "WARN",
            show_default=True,
            help="Select logger option, from INFO, WARN or DEBUG.",
            ),
        outdir: Path = typer.Option(
            Path.cwd()/"reports/videos",
            help=("Output directory for the results of the monitor."+
                  "[default: ./reports/videos]"),
            ),
        ):
    """Play an OpenAI Gym game using a DQL agent previously trained.

    This module creates an OpenAI Gym environment and executes actions
    as dictated from a learned policy. The game is rendered on screen,
    and the results can be recorded. The default game is "PongNoFrameskip-v4".
    """
    typer.echo(f"Playing {game} with a trained agent.")

    # Set the logger level
    if logger == "INFO":
        gym.logger.set_level(gym.logger.INFO)
    elif logger == "DEBUG":
        gym.logger.set_level(gym.logger.DEBUG)
    elif logger == "WARN":
        gym.logger.set_level(gym.logger.WARN)

    # Make and wrap the environment
    env = wrappers.make_env(game)
    if monitor:
        env = gymwr.Monitor(env, directory=outdir, force=True)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(model, map_location=torch.device("cpu")))
    rendered = env.render(mode='rgb_array')

    # Set size of game window
    video_size=[rendered.shape[1], rendered.shape[0]]
    video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
    screen = pygame.display.set_mode(video_size)

    done = False

    for ep in range(episodes):
        typer.echo(f"Starting episode {ep}.")
        total_reward = 0
        state = env.reset()
        state_count = 0
        action_counter = collections.Counter()
        while True:
            start_ts = time.time()
            if not monitor: env.render()
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            action_counter[action] += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if verbose:
                typer.echo(f"{action} {reward} {state}")
            if done:
                typer.echo(f"Game reached end-state in frame {state_count}.")
                break
            delta = 1/fps - (time.time() - start_ts)
            if (delta > 0) and not monitor:
                time.sleep(delta)
            state_count += 1

    env.env.close()

if __name__ == "__main__":
    typer.run(play_gym_model)
