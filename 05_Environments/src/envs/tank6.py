"""
Gym Environment for Tank Saturdays.
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from IPython.display import clear_output
import matplotlib.pyplot as plt
import typer
from collections import namedtuple, defaultdict
from recordclass import recordclass

class TankSaturdays(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }


    def __init__(self):

        BATTLEFIELD_SIDE = 50
        TANK_SIDE = 5
        GAS = 1000
        CARTRIDGE = 100

        self.bf_side = BATTLEFIELD_SIDE
        self.bf_size = np.array([BATTLEFIELD_SIDE, BATTLEFIELD_SIDE])
        self.tank_side = TANK_SIDE
        self.tank_size = np.array([TANK_SIDE, TANK_SIDE])
        self.pad = self.tank_side//2
        self.bullet_speed = 3
        self.n_walls = 10
        self.width_walls = 2
        self.length_walls = (5, 16)
        self.gas = GAS
        self.cartridge = CARTRIDGE

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(np.prod(self.bf_size))

        self.actions = {
            "idle": 0,
            "move_up": 1,
            "move_right": 2,
            "move_down": 3,
            "move_left": 4,
            "shoot_up": 5,
            "shoot_right": 6,
            "shoot_down": 7,
            "shoot_left": 8,
            }

        self.dv_tuple = namedtuple('Velocities', ['dx', 'dy'])
        self.v_actions = defaultdict(lambda x: self.dv_tuple(0, 0))
        self.v_actions[1] = self.dv_tuple(0, -1)
        self.v_actions[2] = self.dv_tuple(1, 0)
        self.v_actions[3] = self.dv_tuple(0, 1)
        self.v_actions[4] = self.dv_tuple(-1, 0)

        self.bullet = recordclass('Bullet', ['x', 'y', 'dx', 'dy'])
        self.wall = namedtuple('Wall', ['x0', 'y0', 'x1', 'y1'])
        self.tank = recordclass('Tank', ['x', 'y', 'gas', 'bullets'])

        self.seed()
        self.reset()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(
            seed if seed is not None else np.random.seed())
        return [seed]

    def step(self, action_1, action_2=None):
        """
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
        """

        # move bullets
        self._move_bullets()

        # Check collisions between tanks and walls for given action
        #action, action_2 = self._wall_collisions(action_1, action_2)
        #
        # self.black.x += self.v_actions[action_1].dx
        # self.black.y += self.v_actions[action_1].dy

        if action_1 == self.actions["idle"]:
            pass
        elif action_1 == self.actions["move_up"]:
            self.black.y = max(self.pad, self.black.y-1)
        elif action_1 == self.actions["move_right"]:
            self.black.x = min(self.bf_side-self.pad, self.black.x+1)
        elif action_1 == self.actions["move_down"]:
            self.black.y = min(self.bf_side-self.pad, self.black.y+1)
        elif action_1 == self.actions["move_left"]:
            self.black.x = max(self.pad, self.black.x-1)

        if action_1 == self.actions["shoot_up"]:
            self.bullets.append(self.bullet(self.black.x, self.black.y-self.pad-1,
                                            0, -self.bullet_speed))
        elif action_1 == self.actions["shoot_right"]:
            self.bullets.append(self.bullet(self.black.x+self.pad+1, self.black.y,
                                            self.bullet_speed, 0))
        elif action_1 == self.actions["shoot_down"]:
            self.bullets.append(self.bullet(self.black.x, self.black.y+self.pad+1,
                                            0, self.bullet_speed))
        elif action_1 == self.actions["shoot_left"]:
            self.bullets.append(self.bullet(self.black.x-self.pad-1, self.black.y,
                                            -self.bullet_speed, 0))


        # remove bullets out of battlefield
        self._remove_bullets()
        # check collisions (bullet - tank)

        # check collisions ( bullet-bullet)

        reward = 0
        done = False


        return self._get_obs(), reward, done, {}

    def reset(self):
        #self.x = self.np_random.randint(low = self.pad,
        #                                high = self.bf_side-self.pad)
        #self.y = self.np_random.randint(low = self.pad,
        #                                high = self.bf_side-self.pad)

        self.black = self.tank(self.pad, self.pad, self.gas, self.cartridge)
        self.white = self.tank(self.bf_side-self.pad, self.bf_side-self.pad,
            self.gas, self.cartridge)

        self.bullets = []

        self.wall_m = np.zeros(self.bf_size)
        self.walls = []
        for _ in range(self.n_walls):
            dir = self.np_random.randint(2)
            len_ = self.np_random.randint(*self.length_walls)

            x0 = self.np_random.randint(low = self.tank_side,
                                    high = self.bf_side-self.tank_side-len_)
            x1 = x0 + len_

            y0 = self.np_random.randint(low = self.tank_side,
                                    high = self.bf_side-self.tank_side-self.width_walls)
            y1 = y0 + self.width_walls

            if dir == 1:
                x0, x1, y0, y1 = y0, y1, x0, x1

            self.walls.append(self.wall(x0, y0, x1, y1))
            self.wall_m[x0:x1, y0:y1] = 1


        self.state = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _move_bullets(self):
        for i, bullet in enumerate(self.bullets):
            self.bullets[i].x += bullet.dx
            self.bullets[i].y += bullet.dy


    def _remove_bullets(self):
        self.bullets = list(filter(lambda b: (b.x>=0) and (b.x<self.bf_side), \
                                         self.bullets))

        self.bullets = list(filter(lambda b: (b.y>=0) and (b.y<self.bf_side), \
                                         self.bullets))

    def _wall_collisions(self, tank, dx, dy):
        tank_m = np.zeros(self.bf_size)
        tank_m[tank.x-self.pad + dx:tank.x+self.pad+1 + dx,
               tank.y-self.pad + dy:tank.y+self.pad+1 + dy] = 1


        return None

    def render(self, mode='console'):

        if mode == 'rgb_array':

            pass

        if mode == "console":

            self.render_m = np.full(self.bf_size, '·')
            self.render_m[self.black.x-self.pad:self.black.x+self.pad+1,
                          self.black.y-self.pad:self.black.y+self.pad+1] = '■'
            for bullet in self.bullets:
                if np.abs(bullet.dx) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '|'
                elif np.abs(bullet.dy) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '—'

            for wall in self.walls:
                self.render_m[wall.x0:wall.x1, wall.y0:wall.y1] = 'X'

            for row in self.render_m.tolist():
                print(" ".join(row))
            print(len(self.bullets))
            typer.clear()

    def get_keys_to_action(self):
        return {(37,): 1, (38,): 2, (39,): 3, (40,): 4}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
