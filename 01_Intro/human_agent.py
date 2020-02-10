#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import pygame
from pygame.locals import VIDEORESIZE
import gym
import pandas as pd
from gym import wrappers
from pathlib import Path

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(
        0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def play_gym_human(
        game: str = typer.Argument("Pong-ram-v4"),
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
        fps: int = typer.Option(
            30,
            show_default=True,
            help="Frames per second for the game",
            ),
        zoom: float = typer.Option(
            1.0,
            show_default=True,
            help="Zoom level on the game window",
            ),
        verbose: bool = typer.Option(
            False,
            show_default=True,
            help="Print action, reward and observation at every step",
            ),
        record: bool = typer.Option(
            False,
            show_default=True,
            help="Record the observations, actions and rewards in a DataFrame"
            ),
        logger: str = typer.Option(
            "WARN",
            show_default=True,
            help="Select logger option, from INFO, WARN or DEBUG",
            ),
        outdir: Path = typer.Option(
            Path.cwd()/"data",
            help=("Output directory for the results of the recorder"+
                  "[default: ./data]"),
            ),
        filename: str = typer.Option(
            None,
            help=("Name for the Pickle file with the gameplay recording"),
            ),
        ):
    """Play an OpenAI Gym game yourself.

    This module creates an OpenAI Gym environment and allows you to play it
    using your keyboard. Your actions can be recorded and stored,
    """
    typer.echo(f"Playing {game} with a human player.")

    # Set the logger level
    if logger == "INFO":
        gym.logger.set_level(gym.logger.INFO)
    elif logger == "DEBUG":
        gym.logger.set_level(gym.logger.DEBUG)
    elif logger == "WARN":
        gym.logger.set_level(gym.logger.WARN)


    # Make and seed the environment
    env = gym.make(game)
    rendered=env.render(mode='rgb_array')
    obs = env.reset()

    # Get the key to actions mapping for the environment
    if hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        keys_to_action = env.unwrapped.get_keys_to_action()
    else:
        assert False, env.spec.id + " does not have explicit key to action " +\
                          "mapping, please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    # Set size of game window
    video_size=[rendered.shape[1], rendered.shape[0]]
    video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
    screen = pygame.display.set_mode(video_size)

    pressed_keys = []
    transpose = True
    running = True
    env_done = False

    clock = pygame.time.Clock()

    if record:
        #ob_features = ["ob_"+str(i) for i in range(obs.size)]
        record_df = pd.DataFrame(columns=["episode", "time", "observation",
                                          "action", "reward"])

    ep = 0
    time = 0
    r = 0
    while running:
        if env_done:  # Natural end of a simulation, game is resetted
            typer.echo(f"Game reached end-state in frame {time}.")
            ep += 1
            time = 0
            env_done = False
            obs = env.reset()
        elif time >= frame_limit:  # Set frame limit reached, game is resetted
            typer.echo(f"Frame limit of {frame_limit} reached.")
            ep += 1
            time = 0
            env_done = False
            obs = env.reset()
        elif ep >= episodes:  # Preset episodes player, game is closed
            typer.echo(f"Maximum number of {episodes} episodes reached.")
            running = False
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            prev_obs = obs
            observation, reward, env_done, info = env.step(action)
            if verbose:
                typer.echo(f"{ep} {time} {action} {reward} {observation}")
            if record:
                record_df.loc[r] = [ep, time, observation, action, reward]
                r+=1
            time += 1

        if obs is not None:
            rendered=env.render( mode='rgb_array')
            display_arr(screen, rendered,
                        transpose=transpose,
                        video_size=video_size)

        # Process pygame events
        for event in pygame.event.get():
            # Test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                typer.echo(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
    typer.echo(f"Game exited.")

    if record:
        record_df.set_index(["episode", "time"], inplace=True)
        if filename is not None:
            file_path = outdir/filename
        else:
            file_path = outdir/f"{game}-human-record.pkl"
        record_df.to_pickle(file_path)
        typer.echo(f"Gameplay recorded into Pickle file at {file_path}")

if __name__ == "__main__":
    typer.run(play_gym_human)
