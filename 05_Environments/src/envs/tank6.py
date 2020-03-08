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

        BATTLEFIELD_SIDE = 20
        TANK_SIDE = 3
        GAS = 100
        CARTRIDGE = 100
        HP = 3
        BULLET_SPEED = 3
        N_WALLS = 10
        WIDTH_WALLS = 2

        self.bf_side = BATTLEFIELD_SIDE
        self.tank_side = TANK_SIDE
        self.gas = GAS
        self.cartridge = CARTRIDGE
        self.HP = HP
        self.bullet_speed = BULLET_SPEED
        self.n_walls = N_WALLS
        self.width_walls = WIDTH_WALLS
        self.length_walls = (self.tank_side, self.bf_side//2)

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
        self.shoot_tuple = namedtuple('Shoot', ['x', 'y', 'dx', 'dy'])

        # Map of actions to tank movement, defaulting to no movement
        self.v_actions = defaultdict(lambda: self.dv_tuple(0, 0))
        self.v_actions[1] = self.dv_tuple(0, -1)
        self.v_actions[2] = self.dv_tuple(1, 0)
        self.v_actions[3] = self.dv_tuple(0, 1)
        self.v_actions[4] = self.dv_tuple(-1, 0)

        # Map of actions to tank shots, defaulting to no shot
        self.s_actions = defaultdict(lambda: None)
        self.s_actions[5] = self.shoot_tuple(
            0, -self.pad-1, 0, -self.bullet_speed)
        self.s_actions[6] = self.shoot_tuple(
            self.pad+1, 0, self.bullet_speed, 0)
        self.s_actions[7] = self.shoot_tuple(
            0, self.pad+1, 0, self.bullet_speed)
        self.s_actions[8] = self.shoot_tuple(
            -self.pad-1, 0, -self.bullet_speed, 0)


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

        # Put tanks in opposite corners, with max gas, bullets and life
        self.black = self.tank(self.pad, self.pad,
            self.gas, self.cartridge, self.HP)
        self.white = self.tank(self.bf_side-self.pad-1, self.bf_side-self.pad-1,
            self.gas, self.cartridge, self.HP)

        # Reset bullets in play
        self.bullets = []
        self.new_bullets = []

        # Put walls in random locations in the center of the battlefield
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

            if dir == 1:  # If vertical wall, transpose dimensions
                x0, x1, y0, y1 = y0, y1, x0, x1

            self.walls.append(self.wall(x0, y0, x1, y1))
            self.wall_m[x0:x1, y0:y1] = 1

        self.state = None
        return self._get_obs()

    def step(self, action_b, action_w=None):

        self.last_action_b = action_b
        self.last_action_w = action_w

        # Get tank displacement from actions
        action_b_v = self.v_actions[action_b]
        action_w_v = self.v_actions[action_w]

        # Get tank shots from actions
        action_b_s = self.s_actions[action_b]
        action_w_s = self.s_actions[action_w]

        # Suppress movement when it would collide or if there is no gas left
        if self._wall_collisions(self.black, action_b_v) or self.black.gas == 0:
            action_b_v = self.v_actions[0]
        if self._wall_collisions(self.white, action_w_v) or self.white.gas == 0:
            action_w_v = self.v_actions[0]

        # Suppress shooting if cartridge is empty
        if self.black.cartridge == 0:
            action_b_s = None
        if self.white.cartridge == 0:
            action_w_s = None

        # Move tanks
        self._move_tanks(action_b_v, action_w_v)

        # Check collisions between tanks and end game as draw if they collided
        if self._tank_collision():
            return self._get_obs(), 0, True, {}

        # Substract gas from tanks if they moved
        self.black.gas -= (1 if action_b_v != self.v_actions[0] else 0)
        self.white.gas -= (1 if action_w_v != self.v_actions[0] else 0)

        # Move bullets
        self._move_bullets()

        # Shoot new bullets
        self._shoot_bullets(action_b_s, action_w_s)

        # Substract bullets from cartridge
        self.black.cartridge -= (1 if action_b_s is not None else 0)
        self.white.cartridge -= (1 if action_w_s is not None else 0)

        # Add new bullets to record of bullets in flight
        self.bullets.extend(self.new_bullets)

        # Check bullet-bullet collitions and remove collided bullets
        self._bullet_bullet_collisions()

        # Remove bullets out of battlefield
        self._remove_bullets()
        # check collisions (bullet - tank)

        # check collisions ( bullet-bullet)

        reward = 0
        done = False


        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state

    def _move_tanks(self, action_b_v, action_w_v):
        self.black.x += action_b_v.dx
        self.black.y += action_b_v.dy
        self.white.x += action_w_v.dx
        self.white.y += action_w_v.dy

    def _move_bullets(self):
        """Displace all bullets in flight."""
        for i, bullet in enumerate(self.bullets):
            self.bullets[i].x += bullet.dx
            self.bullets[i].y += bullet.dy
        # bullet_collision_space = np.zeros(self.bf_size)
        # to_delete = []
        # for i, bullet in enumerate(self.bullets):
        #     for _ in range(self.bullet_speed):
        #         self.bullets[i].x += np.sign(bullet.dx)
        #         self.bullets[i].y += np.sign(bullet.dy)
        #         if ((self.bullets[i].x<0) or (self.bullets[i].x>=self.bf_side)
        #          or (self.bullets[i].y<0) or (self.bullets[i].y>=self.bf_side)):
        #             to_delete.append(i)
        #             break
        #         if self.wall_m[self.bullets[i].x, self.bullets[i].y] == 1:
        #             to_delete.append(i)
        #             break
        #         bullet_collision_space[self.bullets[i].x, self.bullets[i].y] = 1
        # new_bullets = []
        # for i, bullet in enumerate(self.bullets):
        #     if i not in to_delete:
        #         new_bullets.append(bullet)
        # self.bullets = new_bullets
        # return bullet_collision_space

    def _shoot_bullets(self, action_b_s, action_w_s):
        """Check if any tank tried to shoot and create the bullet if so."""
        self.new_bullets = []
        if action_b_s is not None:
            self.new_bullets.append(self.bullet(
                self.black.x+action_b_s.x, self.black.y+action_b_s.y,
                action_b_s.dx, action_b_s.dy))
        if action_w_s is not None:
            self.new_bullets.append(self.bullet(
                self.white.x+action_w_s.x, self.white.y+action_w_s.y,
                action_w_s.dx, action_w_s.dy))

    def _remove_bullets(self):
        """Removes bullets that went out of the battlefield."""
        self.bullets = list(filter(lambda b: (b.x>=0) and (b.x<self.bf_side),
                                         self.bullets))
        self.bullets = list(filter(lambda b: (b.y>=0) and (b.y<self.bf_side),
                                         self.bullets))

    def _wall_collisions(self, tank, action_v):
        """Returns True if the tank would collide with a wall or border if it
        acted on given action.
        """
        dx = action_v.dx
        dy = action_v.dy

        if (dx==0) and (dy==0):  # No need to check this case
            return False

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

        return collisions > 0

    def _bullet_bullet_collisions(self):
        keep_bullets = set()
        for i, bullet_1 in enumerate(self.bullets):
            collision = False
            for j, bullet_2 in enumerate(self.bullets):
                if (bullet_1.x == bullet_2.x) and (bullet_1.y == bullet_2.y):
                    if j != i:
                        collision = True
            if not collision:
                keep_bullets.add(i)
        self.bullets = [self.bullets[i] for i in keep_bullets]


    def _tank_collision(self):
        """Returns True if the tanks collide in the current position."""
        x_collide = ((self.white.x+self.pad+1 > self.black.x-self.pad) and
                     (self.black.x+self.pad+1 > self.white.x-self.pad))
        y_collide = ((self.white.y+self.pad+1 > self.black.y-self.pad) and
                     (self.black.y+self.pad+1 > self.white.y-self.pad))
        return x_collide and y_collide

    def render(self, mode='console'):

        if mode == 'rgb_array':

            pass

        if mode == "console":

            # Background
            self.render_m = np.full(self.bf_size, '·')

            # Tanks
            self.render_m[self.black.x-self.pad:self.black.x+self.pad+1,
                          self.black.y-self.pad:self.black.y+self.pad+1] = '■'
            self.render_m[self.white.x-self.pad:self.white.x+self.pad+1,
                          self.white.y-self.pad:self.white.y+self.pad+1] = '□'

            # Bullets
            for bullet in self.bullets:
                if np.abs(bullet.dx) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '|'
                elif np.abs(bullet.dy) == self.bullet_speed:
                    self.render_m[bullet.x, bullet.y] = '—'

            # Walls
            for wall in self.walls:
                self.render_m[wall.x0:wall.x1, wall.y0:wall.y1] = 'X'

            # Print to console
            for row in self.render_m.tolist():
                print(" ".join(row))
            print(f"Gas left: {self.black.gas};   "
                  f"Bullets left: {self.black.cartridge}")
            typer.clear()


    def get_keys_to_action(self):
        return {(37,): 1, (38,): 2, (39,): 3, (40,): 4}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
