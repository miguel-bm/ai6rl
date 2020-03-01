#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import time
import gym
import torch
import collections
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pygame
from pygame.locals import VIDEORESIZE

import gym.wrappers as gymwr

from models import wrappers
from models import dqn_model


class ModelAgent(object):
    """Acts acording to loaded Pytorch model."""
    def __init__(self, env, model_path):
        # Load the model and set it to inference.
        self.model = dqn_model.DQN(env.observation_space.shape,
                                   env.action_space.n)
        self.model.load_state_dict(torch.load(model_path,
            map_location=torch.device("cpu")))
        self.model.eval()

    def act(self, observation):
        obs = torch.tensor(np.array([observation], copy=False))
        q_values = self.model(obs).data.numpy()[0]
        action = np.argmax(q_values)
        return action

    def value(self, observation, action):
        obs = torch.tensor(np.array([observation], copy=False))
        q_values = self.model(obs).data.numpy()[0]
        return q_values[action]


class ModelEmbeddings(object):
    """Returns the values from the last hidden layer of a model."""
    def __init__(self, env, model_path):
        # Load the model and set it to inference.
        self.model = dqn_model.DQN_emb(env.observation_space.shape,
                                       env.action_space.n)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        del state_dict["fc.2.weight"]
        del state_dict["fc.2.bias"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def embeddings(self, observation):
        obs = torch.tensor(np.array([observation], copy=False))
        embeddings = self.model(obs).data.numpy()[0]
        return embeddings

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
        verbose: bool = typer.Option(
            False,
            show_default=True,
            help="Print action, reward and observation at every step.",
            ),
        plot_tsne: bool = typer.Option(
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

    # Set up the agent
    agent = ModelAgent(env, model)
    if plot_tsne:
        embnet = ModelEmbeddings(env, model)
        embeddings = []
        actions = []
        rewards = []
        q_values = []


    # Make it so you can zoom on the window
    rendered = env.render(mode='rgb_array')
    zoom = 1.0
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
            action = agent.act(state)
            action_counter[action] += 1
            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            if plot_tsne:
                embeddings.append(embnet.embeddings(state))
                actions.append(action)
                rewards.append(reward)
                q_values.append(agent.value(state, action))
            if verbose:
                typer.echo(f"{action} {reward} {state}")
            if done:
                typer.echo(f"Game reached end-state in frame {state_count}, "
                           f"achieving a total reward of {total_reward}.")
                break

            # Stop for a bit to make gameplay slower in the renderer
            delta = 1/fps - (time.time() - start_ts)
            if (delta > 0) and not monitor:
                time.sleep(delta)

            state_count += 1
            state = new_state

    env.env.close()

    if plot_tsne:
        import matplotlib as mpl
        typer.echo("Performing t-SNE embedding.")
        X_tsne = TSNE(n_components=2).fit_transform(np.array(embeddings))
        data = pd.DataFrame(data = {"tsne0": X_tsne[:, 0],
                                    "tsne1": X_tsne[:, 1],
                                    "action": actions,
                                    "reward": rewards,
                                    "q_value": q_values,
                                    })

        meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        markers = ["o", "x", "^", "v", "^", "v"]
        data["Action"] = data["action"].apply(lambda x: meanings[x])

        cmap = mpl.colors.Colormap("viridis")
        norm = mpl.colors.Normalize(vmin=0, vmax=max(data["q_value"]))
        scmap = mpl.cm.ScalarMappable(norm, cmap)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Two-dimensional t-SNE embedding of the representations\n"
                     "in the last hidden layer assigned by DQN to game states")
        for marker, action in zip(markers, meanings):
            a_data = data[data["Action"]==action]
            plt.scatter(a_data["tsne0"], a_data["tsne1"], c=a_data["q_value"],
                        cmap="viridis", norm=norm, alpha=0.7, s=50.0,
                        marker=marker)
        plt.savefig(Path.cwd()/"reports/figures/tsne_q_values.png")
        plt.close()


if __name__ == "__main__":
    typer.run(play_gym_model)
