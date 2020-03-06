"""
Gym Environment that reproduces the well known smart phone game "2048".
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from IPython.display import clear_output
import matplotlib.pyplot as plt
import typer


class g2048(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.board_size = np.array([4, 4])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.prod(self.board_size))

        self.seed()
        self.reset()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(
            seed if seed is not None else np.random.seed())
        return [seed]

    def step(self, move):
        assert int(move) in range(4)
        old_state = self.state.copy()
        old_full = np.sum(old_state == 0)

        rotated_state = np.rot90(old_state, move)
        new_rotated_state = []
        for i, row in enumerate(rotated_state):
            zero_counter = 0
            new_row = row.copy()
            for j, value in enumerate(row):
                if value == 0:
                    zero_counter += 1
                else:
                    for k, kvalue in enumerate(new_row[:j]):
                        if kvalue == 0:
                            new_row[k] = value
                            new_row[j] = 0
                            break
                        elif kvalue == value:
                            new_row[k] = new_row[k] + 1
                            new_row[j] = 0
                            break
            new_rotated_state.append(new_row)
        self.state = np.rot90(np.array(new_rotated_state), 4-move)

        if (self.state == old_state).all():
            reward = 0
        else:
            reward, _ = self._put_piece()

        if np.sum(self.state == 0) == 0:  # No empty positions
            done = True  # TBD: improve done logic, not fully correct ATM
        else:
            done = False

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = np.zeros(self.board_size, dtype=np.int32)
        self._put_piece()
        self._put_piece()
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _get_pos(self):
        """Select a random position in the state where there is no piece."""
        zero_pos = list(np.array(np.where(self.state == 0)).T)
        return zero_pos[self.np_random.randint(len(zero_pos))]

    def _get_piece(self):
        """Select a piece value to be positioned."""
        return 1 if self.np_random.uniform() < 0.9 else 2

    def _put_piece(self):
        """Put piece with random value in empty position."""
        position = self._get_pos()
        value = self._get_piece()
        self.state[position[0]][position[1]] = value
        return value, position

    def render(self, mode='rgb_array'):

        if mode == 'rgb_array':

            tile_color_map = {
                2: (255, 255, 0),
                4: (255, 0, 0),
                8: (255, 0, 255),
                16: (0, 0, 255),
                32: (0, 255, 255),
                64: (0, 255, 0),
                128: (128, 128, 0),
                256: (128, 0, 0),
                512: (128, 0, 128),
                1024: (0, 0, 128),
                2048: (0, 128, 128),
                4096: (0, 128, 0),
                8192: (0, 0, 0),
            }

            ts = 100
            bs = self.board_size * ts

            import pyglet
            from gym.envs.classic_control import rendering
            class DrawText:
                def __init__(self, label:pyglet.text.Label):
                    self.label=label
                def render(self):
                    self.label.draw()

            if self.viewer is None:
                self.viewer = rendering.Viewer(bs[1], bs[0])

            self.viewer = rendering.Viewer(bs[1], bs[0])
            self.viewer.set_bounds(0, bs[1], 0, bs[0])


            for row in range(self.board_size[0]):
                for col in range(self.board_size[1]):
                    value = 2**int(self.state[row, col])
                    if value == 1:
                        continue
                    rectangle_coords = [
                        (ts*(col+0.05), ts*(row+0.05)),
                        (ts*(col+0.95), ts*(row+0.05)),
                        (ts*(col+0.95), ts*(row+0.95)),
                        (ts*(col+0.05), ts*(row+0.95)),
                        ]
                    color = tile_color_map[value]

                    text = str(value)
                    label = pyglet.text.Label(text, font_size=25,
                        x=ts*(col+0.5), y=ts*(row+0.5),
                        anchor_x='center', anchor_y='center',
                        color=(0, 0, 0, 255))
                    label.draw()

                    #self.viewer.add_geom(DrawText(label))

                    self.viewer.draw_polygon(rectangle_coords, color=color)

            return self.viewer.render(return_rgb_array = mode=='rgb_array')

        if mode == "console":
            typer.clear()
            cols = self.board_size[1]
            rows = self.board_size[0]
            row_divider = "-----".join(["+"]*(cols+1))
            row_padder = "     ".join(["|"]*(cols+1))
            values = (2**self.state).tolist()
            str_values = [[str(value).rjust(3).ljust(5) if value != 1
                          else "     "
                          for value in row]
                           for row in values]

            typer.echo(row_divider)
            for row in str_values:
                typer.echo(row_padder)
                typer.echo("|"+"|".join(row)+"|")
                typer.echo(row_padder)
                typer.echo(row_divider)

    def get_keys_to_action(self):
        return {(37,): 0, (38,): 1, (39,): 2, (40,): 3} # Control with arrows

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
