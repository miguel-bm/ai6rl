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


class ImitatorAgent(object):
    """Acts acording to loaded Pytorch model."""
    def __init__(self, model_path, action_space):
        self.action_space = sorted(list(action_space))
        self.action_decoding = {i: x for i, x in enumerate(action_space)}

        state_dict = torch.load(model_path)
        insize = list(state_dict.values())[0].shape[1]
        densesize = list(state_dict.values())[0].shape[0]
        outsize = list(state_dict.values())[-1].shape[0]
        self.model = DenseNN(insize, outsize, densesize)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, observation, reward, done):
        obs = torch.tensor(np.expand_dims(observation, axis=0),
                           dtype=torch.float)
        output = torch.nn.Softmax(dim=1)(self.model(obs)).data.numpy().flatten()
        coded_action = np.random.choice(len(output), 1, p=output)
        return self.action_decoding[int(coded_action)]


def play_gym_model(
        game: str = typer.Argument("Pong-ram-v4"),
        model: Path = typer.Argument("models/nn_imitator.pt"),
        episodes: int = typer.Option(
            1,
            show_default=True,
            help="Number of runs of the environment to simulate consecutively",
            ),
        frame_limit: int = typer.Option(
            10000,
            show_default=True,
            help="Maximum number of steps to execute for each episode",
            ),
        fps: int = typer.Option(
            30,
            show_default=True,
            help="Frames per second (actually an upper bound)"
            ),
        verbose: bool = typer.Option(
            False,
            show_default=True,
            help="Print action, reward and observation at every step",
            ),
        monitor: bool = typer.Option(
            False,
            show_default=True,
            help="Activate a monitor to record a video of the results"
            ),
        logger: str = typer.Option(
            "WARN",
            show_default=True,
            help="Select logger option, from INFO, WARN or DEBUG ",
            ),
        outdir: Path = typer.Option(
            Path.cwd()/"model_results",
            help=("Output directory for the results of the monitor"+
                  "[default: ./model_results]"),
            ),
        ):
    """Play an OpenAI Gym game using a random agent.

    This module creates an OpenAI Gym environment and executes random actions
    from its action space, sampled uniformly. The game is rendered on screen,
    and the results can be recorded. The default game is "CartPole-v1".
    """
    typer.echo(f"Playing {game} with a random agent.")

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

    agent = ImitatorAgent(model, [0, 2, 3])  # Hack, only works in pong, hard
                                             # implement because gym is stupid
    reward = 0
    done = False

    for ep in range(episodes):
        typer.echo(f"Starting episode {ep+1}.")
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
