#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
import imageio
from pathlib import Path
from collections import defaultdict, Counter

class TabularQLearningAgent():
    def __init__(self, env_name: str="FrozenLake-v0",
                       discount_factor: float=0.9,
                       learning_rate: float=0.2,
                       value_table: dict={}):
        self.env = gym.make(env_name)
        self.df = discount_factor
        self.lr = learning_rate
        self.state = self.env.reset()
        self.values = defaultdict(float, value_table)  # Table of Q values

    def random_step(self):
        """Play randomly in the environment for one step, resetting if needed.
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else new_state
        return old_state, action, reward, new_state

    def update_value(self, state, action, reward, next_state):
        """Apply the Bellman equation and blending in order to update the
        learned state-action values.
        """
        best_value, _ = self.best_value_and_action(next_state)
        new_value = reward + self.df * best_value
        old_value = self.values[(state, action)]
        self.values[(state, action)] = ((1-self.lr) * old_value +
                                           self.lr  * new_value)

    def best_value_and_action(self, state):
        """Considers every action available and selects the one with the
        greatest Q value.
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_action is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def play_episode(self, env, populate_tables=False):
        """Play one episode of the environment by selecting actions through the
        current knowledge accumulated in the value table.
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            if populate_tables:
                self.update_value(self, state, action, reward, next_state)
            if done:
                break
            state = new_state
        return total_reward

    def get_value_table(self):
        return self.values.copy()


def train_TQL(
    environment: str = typer.Argument("FrozenLake-v0"),
    learning_rate: float = typer.Option(
        0.2,
        show_default=True,
        help = "Rate of blending for updating old Q values with new ones.",
        ),
    discount_factor: float = typer.Option(
        0.9,
        show_default=True,
        help = "Decrease reward of later steps in a trajectory.",
        ),
    batch_size: int = typer.Option(
        20,
        show_default=True,
        help = "Number of episodes for testing policy at each iteration.",
        ),
    max_iter: int = typer.Option(
        10000000,
        show_default=True,
        help = "Maximum number of iterations to train the model for.",
        ),
    reward_objective: float = typer.Option(
        0.8,
        show_default=True,
        help = "Objective of reward aaveraged over a test batch to reach.",
        ),
    save: bool = typer.Option(
        True,
        show_default=True,
        help = "Save the trained model to disk.",
        ),
    populate_in_test: bool = typer.Option(
        False,
        show_default=True,
        help = "Use data from testing to populate the tables (EXPERIMENTAL).",
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
    """Train a Tabular Q-learning model by populating value tables.

    The model will first randomly explore the environment for one step, then
    apply the Bellman equation for updating a state-action value, using a
    learning rate for blending the old value with the new one, and then test
    the environment for a certain number of episodes to check whether the
    convergence conditions are met (average reward). Repeat until done.

    Args:
        environment (str): name of the Gym environment to train the model on.

    Returns:
        A defaultdict containing the state-action value table of the
        environment.
    """

    typer.echo(f"Training a tabular Q-learning RL model for {environment}")

    # Create environment for testing
    env = gym.make(environment)

    # Initiate Agent
    agent = TabularQLearningAgent(environment, learning_rate, discount_factor)

    best_reward = 0.0
    for iter in range(max_iter):
        # Explore environment and populate tables by playing randomly
        state, action, reward, next_state = agent.random_step()

        # Perform the update on the tabular Q values from the step
        agent.update_value(state, action, reward, next_state)

        # Use the table as a policy for playing a batch of test episodes
        total_reward = 0.0
        for episode in range(batch_size):
            total_reward += agent.play_episode(env)
        average_reward = total_reward / batch_size

        if average_reward > best_reward:
            typer.echo(f"Best reward updated {best_reward:6.3f} -> "+
                       f"{average_reward:6.3f} at iteration {iter:6d}")
            best_reward = average_reward

        # Chech for the objective
        if average_reward > reward_objective:
            typer.echo("Reward objective reached!")
            break

    typer.echo(f"Process stoped after {iter+1} interations.")

    # Save the model as a pickle file, which will hold the dict with the QF
    if save:
        if save_name is None:
            save_name = f"TQL_table_{environment}.pkl"
        elif save_name[-4] != ".pkl":
            save_name += ".pkl"
        typer.echo(f"Saving model to {outdir/save_name}")
        with open(outdir/save_name, 'wb') as handle:
            pickle.dump(agent.get_value_table(),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    return agent.get_value_table()


if __name__ == "__main__":
    typer.run(train_TQL)
