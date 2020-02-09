#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import gym
from gym import wrappers
from pathlib import Path


class RandomAgent(object):
    """Selects random action uniformly from the action space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def play_gym(
        game: str = typer.Argument("CartPole-v1"),
        episodes: int = typer.Option(
            1,
            show_default=True,
            help="Number of runs of the environment to simulate consecutively",
            ),
        frame_limit: int = typer.Option(
            1000,
            show_default=True,
            help="Maximum number of steps to execute for each episode",
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
            Path.cwd()/"random_agent_results",
            help=("Output directory for the results of the monitor"+
                  "[default: ./random_agent_results]"),
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

    agent = RandomAgent(env.action_space)
    reward = 0
    done = False

    for ep in range(episodes):
        typer.echo(f"Starting episode {ep+1}.")
        ob = env.reset()
        state_count = 0
        while True:
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
    typer.run(play_gym)
