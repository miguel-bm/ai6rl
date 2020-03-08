#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import gym
import pandas as pd
from gym import wrappers
import pygame
from pathlib import Path
from envs.tank6 import TankSaturdays
import termios, fcntl, sys, os
import time


def play_2048_human():
    """Play an OpenAI 2048 game yourself.

    This module creates an OpenAI Gym environment and allows you to play it
    using your keyboard.
    """
    typer.echo(f"Playing 2048 with a human player.")


    # Make and seed the environment
    env = TankSaturdays()
    rendered=env.render(mode='console')
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

    pressed_keys = []
    transpose = True
    running = True
    env_done = False

    turn = 0
    total_reward = 0
    while running:
        env.render("console")

        # Process events
        fd = sys.stdin.fileno()

        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

        if env_done:  # Natural end of a simulation, game is resetted
            typer.echo(f"Game reached end-state in turn {turn}.")
            running = False
            break

        try:
            while 1:
                try:
                    c = sys.stdin.read(1)
                    if c:
                        if str(c) == "A":
                            action = 4
                            break
                        elif str(c) == "B":
                            action = 2
                            break
                        elif str(c) == "C":
                            action = 3
                            break
                        elif str(c) == "D":
                            action = 1
                            break
                        elif str(c) == "e":
                            action = 0
                            break
                        elif str(c) == "w":
                            action = 8
                            break
                        elif str(c) == "a":
                            action = 5
                            break
                        elif str(c) == "s":
                            action = 6
                            break
                        elif str(c) == "d":
                            action = 7
                            break
                        elif str(c) == "q":
                            typer.echo(f"Quit game.")
                            running = False
                            break
                except IOError: pass
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        if running == False:
            break

        time.sleep(0.01)

        prev_obs = obs
        observation, reward, env_done, _ = env.step(action, action_w=8)
        turn += 1






if __name__ == "__main__":
    typer.run(play_2048_human)
