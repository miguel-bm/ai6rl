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
        HP = 3
        BULLET_SPEED = 3
        N_WALLS = 10
        WIDTH_WALLS = 2
        LENGTH_WALLS = (5, 16)

        self.bf_side = BATTLEFIELD_SIDE
        self.tank_side = TANK_SIDE
        self.gas = GAS
        self.cartridge = CARTRIDGE
        self.HP = HP
        self.bullet_speed = BULLET_SPEED
        self.n_walls = N_WALLS
        self.width_walls = WIDTH_WALLS
        self.length_walls = LENGTH_WALLS

        self.bf_size = np.array([self.bf_side, self.bf_side])
        self.tank_size = np.array([self.tank_side, self.tank_side])
        self.pad = self.tank_side//2

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

        self.bullet = recordclass('Bullet', ['x', 'y', 'dx', 'dy'])
        self.wall = namedtuple('Wall', ['x0', 'y0', 'x1', 'y1'])
        self.tank = recordclass('Tank', ['x', 'y', 'gas', 'cartridge', 'HP'])
        self.dv_tuple = namedtuple('Velocities', ['dx', 'dy'])

        self.v_actions = defaultdict(lambda: self.dv_tuple(0, 0))
        self.v_actions[1] = self.dv_tuple(0, -1)
        self.v_actions[2] = self.dv_tuple(1, 0)
        self.v_actions[3] = self.dv_tuple(0, 1)
        self.v_actions[4] = self.dv_tuple(-1, 0)


        self.last_action_b = None
        self.last_action_w = None
        self.seed()
        self.reset()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(
            seed if seed is not None else np.random.seed())
        return [seed]

    def reset(self):

        self.black = self.tank(self.pad, self.pad,
            self.gas, self.cartridge, self.HP)
        self.white = self.tank(self.bf_side-self.pad, self.bf_side-self.pad,
            self.gas, self.cartridge, self.HP)

        self.bullets = []

        self.wall_m = np.zeros(self.bf_size)
        self.walls = []
        for _ in range(self.n_walls):
            dir = self.np_random.randint(2)
            length = self.np_random.randint(*self.length_walls)

            x0 = self.np_random.randint(low = self.tank_side,
                                    high = self.bf_side-self.tank_side-length)
            x1 = x0 + length

            y0 = self.np_random.randint(low = self.tank_side,
                    high = self.bf_side-self.tank_side-self.width_walls)
            y1 = y0 + self.width_walls

            if dir == 1:
                x0, x1, y0, y1 = y0, y1, x0, x1

            self.walls.append(self.wall(x0, y0, x1, y1))
            self.wall_m[x0:x1, y0:y1] = 1


        self.state = None
        return self._get_obs()


    def step(self, action_b, action_w=None):

        self.last_action_b = action_b
        self.last_action_w = action_w

        # Move bullets
        self._move_bullets()

        # Get tank displacement from actions
        action_b_v = self.v_actions[action_b]
        action_w_v = self.v_actions[action_w]

        # Check wall collisions and update actions (only if tank tries to move)
        if action_b_v != self.v_actions[0]:
            action_b_v = self._wall_collisions(self.black, action_b_v)
        if action_w_v != self.v_actions[0]:
            action_w_v = self._wall_collisions(self.white, action_w_v)

        # Move tanks
        self.black.x += action_b_v.dx
        self.black.y += action_b_v.dy

        if action_b == self.actions["shoot_up"]:
            self.bullets.append(self.bullet(self.black.x, self.black.y-self.pad-1,
                                            0, -self.bullet_speed))
        elif action_b == self.actions["shoot_right"]:
            self.bullets.append(self.bullet(self.black.x+self.pad+1, self.black.y,
                                            self.bullet_speed, 0))
        elif action_b == self.actions["shoot_down"]:
            self.bullets.append(self.bullet(self.black.x, self.black.y+self.pad+1,
                                            0, self.bullet_speed))
        elif action_b == self.actions["shoot_left"]:
            self.bullets.append(self.bullet(self.black.x-self.pad-1, self.black.y,
                                            -self.bullet_speed, 0))


        # remove bullets out of battlefield
        self._remove_bullets()
        # check collisions (bullet - tank)

        # check collisions ( bullet-bullet)

        reward = 0
        done = False


        return self._get_obs(), reward, done, {}

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

    def _wall_collisions(self, tank, action_v):
        dx = action_v.dx
        dy = action_v.dy

        # Make a matrix with ones in positions occupied by the tank after move
        tank_m = np.zeros(self.bf_size + np.array([2,2]))
        tank_m[tank.x+1-self.pad + dx:tank.x+1+self.pad+1 + dx,
               tank.y+1-self.pad + dy:tank.y+1+self.pad+1 + dy] = 1

        # Make a matrix with ones in positions occupied by walls and borders
        wall_m = np.zeros(self.bf_size + np.array([2,2]))
        wall_m[[0,-1],:], wall_m[:, [0,-1]] = 1, 1  # Walls around battlefield
        for wall in self.walls:
            wall_m[wall.x0+1:wall.x1+1, wall.y0+1:wall.y1+1] = 1

        # Multiply element-wise and sum
        collisions = np.sum(wall_m * tank_m)

        if collisions > 0:
            return self.v_actions[0]  # Set action velocity to idle

        return action_v

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
            if self.last_action_b is not None:
                print(self.last_action_b)
            typer.clear()

    def get_keys_to_action(self):
        return {(37,): 1, (38,): 2, (39,): 3, (40,): 4}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
