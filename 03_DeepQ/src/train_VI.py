#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sns
import imageio
from pathlib import Path
from collections import defaultdict, Counter

class ValueIterationAgent():
    def __init__(self, env_name: str="FrozenLake-v0",
                       discount_factor: float=1.0,
                       value_table: dict={}):
        self.env = gym.make(env_name)
        self.df = discount_factor
        self.state = self.env.reset()
        self.rewards = defaultdict(float)
        self.transits = defaultdict(Counter)
        self.values = defaultdict(float)

    def play_random(self, num_steps):
        """Play randomly in the environment, resetting when necessary, while
        populating the rewards and transits tables.
        """
        for step in range(num_steps):
            # Play random action
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            # Populate tables
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            # Update state or reset environment if done
            self.state = self.env.reset() if done else new_state

    def action_value(self, state, action):
        """Apply the Bellman equation in order to approximate the Q function
        from the information currently in the tables.
        """
        transitions = self.transits[(state, action)]
        total = sum(transitions.values())
        action_value = 0.0
        # Iterate over every target state this state transitions to
        for target_state, count in transitions.items():
            # Calculate the value of the target state
            reward = self.rewards[(state, action, target_state)]
            state_value = reward + self.df * self.values[target_state]
            # The value of the action is the weighted average of the values of
            # the target states
            action_value += (count / total) * state_value
        return action_value

    def select_action(self, state):
        """Considers every action available and selects the one with the
        greatest Q.
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.action_value(state, action)
            if best_action is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        """Iterate over the state (observation) space and approximate the value
        function.
        """
        for state in range(self.env.observation_space.n):
            action_values = [self.action_value(state, action)
                             for action in range(self.env.action_space.n)]
            # The state value is the value of the best action
            self.values[state] = max(action_values)

    def play_episode(self, env, populate_tables=True):
        """Play one episode of the environment by selecting actions through the
        current knowledge accumulated in the value table.
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            if populate_tables:
                self.rewards[(state, action, new_state)] = reward
                self.transits[(state, action)][new_state] += 1
            if done:
                break
            state = new_state
        return total_reward

    def get_value_table(self):
        return self.values.copy()


def draw_vt_heatmap(value_table, title):

    # Turn the value table into a numpy array
    side = int(len(value_table)**0.5)
    values = list()
    for state in range(len(value_table)):
        values.append(value_table[state])
    value_array = np.array(values).reshape((side, side))

    # Draw the heatmap of the state values
    fig, ax = plt.subplots(figsize=(6,5))
    fig.suptitle(title, x=0.15, y= 0.95, fontsize=16, weight="bold", ha="left")
    sns.heatmap(value_array, vmin=0, vmax=1,
                annot=True, fmt="5.3f", ax=ax,
                xticklabels=False, yticklabels=False,
                annot_kws={"fontsize": 14})

    #ax.set_title(title, fontsize=16, weight="bold")

    # Draw the image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def train_VI(
    environment: str = typer.Argument("FrozenLake-v0"),
    exploration_steps: int = typer.Option(
        100,
        show_default=True,
        help = "Number of random steps to explore at each iteration.",
        ),
    discount_factor: float = typer.Option(
        1.0,
        show_default=True,
        help = "Decrease reward of later steps in a trajectory.",
        ),
    batch_size: int = typer.Option(
        20,
        show_default=True,
        help = "Number of episodes for testing policy at each iteration.",
        ),
    max_iter: int = typer.Option(
        1000,
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
    record_gif: bool = typer.Option(
        True,
        show_default=True,
        help = "Make a gif of the evolution of the value table.",
        ),
    gif_name: str = typer.Option(
        None,
        show_default=False,
        help = "Name for the recorded gif.",
        ),
    record_outdir: Path = typer.Option(
        Path.cwd()/"model_results",
        show_default=True,
        help = ("Output directory for the saving the gif " +
                "[default: ./model_results]."),
        ),
    ):
    """Train a Value Iteration model by populating value tables.

    The model will first randomly explore the environment, populating reward
    and transition tables. Then, it will itereate over the various states and
    use the Bellman equation and the accumulated information in order to
    populate a value table approximating the optimal value function.

    Args:
        environment (str): name of the Gym environment to train the model on.

    Returns:
        A defaultdict containing the value table of the environment.
    """

    typer.echo(f"Training a value iteration RL model for {environment}")

    # Create environment for testing
    env = gym.make(environment)

    # Initiate Agent
    agent = ValueIterationAgent(environment,
                                discount_factor)

    if record_gif:
        snapshots = list()

    for iter in range(max_iter):
        # Explore environment and populate tables by playing randomly
        agent.play_random(exploration_steps)

        # Perform the value iteration to populate the value table
        agent.value_iteration()

        # Use the table as a policy for playing a batch of test episodes
        total_reward = 0.0
        for episode in range(batch_size):
            total_reward += agent.play_episode(env)
        average_reward = total_reward / batch_size
        typer.echo(f"Iteration: {iter:4d}     Reward: {average_reward:7.3f}")

        if record_gif:
            value_table = agent.get_value_table()
            snapshots.append(draw_vt_heatmap(value_table,
                title=f"Value Function, iteration {iter:3d}"))

        # Chech for the objective
        if average_reward > reward_objective:
            typer.echo("Reward objective reached!")
            break

    typer.echo(f"Process stoped after {iter} interations.")
    if save:
        typer.echo("Saving model to ")

    if record_gif:
        if gif_name is None:
            gif_name = f"value_function_{environment}.gif"
        elif gif_name[-4:] != ".gif":
            gif_name += ".gif"
        gif_location = record_outdir/gif_name
        kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        imageio.mimsave(gif_location, snapshots, fps=5)

    return agent.get_value_table()


if __name__ == "__main__":
    typer.run(train_VI)
