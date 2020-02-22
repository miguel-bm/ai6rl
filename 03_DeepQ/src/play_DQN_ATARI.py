#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import time
import gym
import torch
import numpy as np
from gym import wrappers
from pathlib import Path
from nn_models import DenseNN


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
        frame_limit: int = typer.Option(
            30000,
            show_default=True,
            help="Maximum number of steps to execute for each episode.",
            ),
        fps: int = typer.Option(
            30,
            show_default=True,
            help="Frames per second (approximately)."
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

    This module creates an OpenAI Gym environment and executes random actions
    from its action space, sampled uniformly. The game is rendered on screen,
    and the results can be recorded. The default game is "CartPole-v1".
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
    env = gym.make(game)
    if monitor:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = ModelAgent(model)  # Hack, only works in pong, hard
                                             # implement because gym is stupid
    reward = 0
    done = False

    for ep in range(episodes):
        typer.echo(f"Starting episode {ep}.")
        ob = env.reset()
        state_count = 0
        while True:
            time.sleep(1/fps)
            state_count += 1
            if not monitor: env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if verbose:
                typer.echo(f"{action} {reward}  {ob}")
            if done:
                typer.echo(f"Game reached end-state in frame {state_count}.")
                break
            elif state_count >= frame_limit:
                typer.echo(f"Frame limit of {frame_limit} reached.")
                break
    env.close()

if __name__ == "__main__":
    typer.run(play_gym_model)
