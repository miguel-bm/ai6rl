"""
Gym Environment for Tank Saturdays.
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
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
        BULLET_SPEED = 5
        IDLE_COST = 1
        MOVE_COST = 2
        N_WALLS = 10
        WIDTH_WALLS = 2
        IDLE_COUNTER = 50

        # Game settings
        self.bf_side = BATTLEFIELD_SIDE
        self.tank_side = TANK_SIDE
        self.gas = GAS
        self.cartridge = CARTRIDGE
        self.HP = HP
        self.bullet_speed = BULLET_SPEED
        self.idle_cost = IDLE_COST
        self.move_cost = MOVE_COST
        self.n_walls = N_WALLS
        self.width_walls = WIDTH_WALLS
        self.length_walls = (self.tank_side, self.bf_side//2)
        self.idle_counter = IDLE_COUNTER

        self.bf_size = np.array([self.bf_side, self.bf_side])
        self.tank_size = np.array([self.tank_side, self.tank_side])
        self.pad = self.tank_side//2

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(np.prod(self.bf_size))

        self.action_map = {  # Not actually used, just for reference
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

        # Named tuples and lists used for recording game object properties
        self.bullet = recordclass('Bullet', ['x', 'y', 'dx', 'dy'])
        self.wall = namedtuple('Wall', ['x0', 'y0', 'x1', 'y1'])
        self.tank = recordclass('Tank', ['x', 'y', 'gas', 'cartridge', 'HP'])
        self.dv_tuple = namedtuple('Velocities', ['dx', 'dy'])
        self.shoot_tuple = namedtuple('Shoot', ['x', 'y', 'dx', 'dy'])
        self.point = namedtuple('Point', ['x', 'y'])

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
        self.viewer = None
        self.hits = list()
        self.image = None
        self.ram = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(
            seed if seed is not None else np.random.seed())
        return [seed]

    def reset(self):
        """Reset the field of play."""

        # Put tanks in opposite corners, with max gas, bullets and life
        self.black = self.tank(self.pad, self.pad,
            self.gas, self.cartridge, self.HP)
        self.white = self.tank(self.bf_side-self.pad-1, self.bf_side-self.pad-1,
            self.gas, self.cartridge, self.HP)

        # Reset bullets in play
        self.bullets = []
        self.new_bullets = []
        self.hits = []  # Merely for rendering purposes

        # Put walls in random locations in the center of the battlefield
        self.wall_m = np.zeros(self.bf_size)
        self.walls = []
        for _ in range(self.n_walls):
            dir = self.np_random.randint(2)
            length = self.np_random.randint(*self.length_walls)
            x0 = self.np_random.randint(low = self.tank_side,
                high = self.bf_side-self.tank_side-length)
            y0 = self.np_random.randint(low = self.tank_side,
                high = self.bf_side-self.tank_side-self.width_walls)
            x1 = x0 + length
            y1 = y0 + self.width_walls
            if dir == 1:  # If vertical wall, transpose dimensions
                x0, x1, y0, y1 = y0, y1, x0, x1

            self.walls.append(self.wall(x0, y0, x1, y1))
            self.wall_m[x0:x1, y0:y1] = 1  # Matrix with ones in wall positions

        self.state = None
        return self._get_obs()

    def step(self, action_b, action_w=None):
        """Executes actions for each tank, simultaneously, and advances time."""

        self.last_action_b = action_b
        self.last_action_w = action_w

        # Get tank displacement from actions
        action_b_v = self.v_actions[action_b]
        action_w_v = self.v_actions[action_w]

        # Get tank shots from actions
        action_b_s = self.s_actions[action_b]
        action_w_s = self.s_actions[action_w]

        # Substract gas from tanks from basic functioning
        self.black.gas -= self.idle_cost
        self.white.gas -= self.idle_cost

        # Suppress movement when it would collide or if there is no gas left
        if self._wall_collision(self.black, action_b_v
            ) or self.black.gas < self.move_cost:
            action_b_v = self.v_actions[0]
        if self._wall_collision(self.white, action_w_v
            ) or self.white.gas < self.move_cost:
            action_w_v = self.v_actions[0]

        # Suppress shooting if cartridge is empty
        if self.black.cartridge <= 0:
            action_b_s = None
        if self.white.cartridge <= 0:
            action_w_s = None

        # Move tanks
        self._move_tanks(action_b_v, action_w_v)

        # Substract gas from tanks if they moved
        if action_b_v != self.v_actions[0]: self.black.gas -= self.move_cost
        if action_w_v != self.v_actions[0]: self.white.gas -= self.move_cost

        # Move bullets
        self._move_bullets()

        # Shoot new bullets
        self._shoot_bullets(action_b_s, action_w_s)

        # Substract bullets from cartridge
        if action_b_s is not None: self.black.cartridge -= 1
        if action_w_s is not None: self.white.cartridge -= 1

        # Check bullet-wall collisions and remove collided bullets
        self._bullet_wall_collisions()  # Before extend, due to bullet speed

        # Check bullet-tank collisions
        self.hits = []  # This is just for rendering, no effect on gameplay
        black_shot = self._bullet_tank_collision(self.black)
        white_shot = self._bullet_tank_collision(self.white)

        # Add new bullets to record of bullets in flight
        self.bullets.extend(self.new_bullets)

        # Check bullet-bullet collisions and remove collided bullets
        self._bullet_bullet_collisions()

        # Remove bullets out of battlefield
        self._remove_bullets()

        # Check collisions between tanks and end game as draw if they collided
        if self._tank_collision():
            return self._get_obs(), 0, True, {}

        # Update HP and end game in case of death(s)
        if black_shot: self.black.HP -= 1
        if white_shot: self.white.HP -= 1
        if (self.black.HP == 0) and (self.white.HP == 0):  # Draw
            return self._get_obs(), 0, True, {}
        elif self.black.HP == 0:  # White tank wins
            return self._get_obs(), -1, True, {}
        elif self.white.HP == 0:  # Black tank wins
            return self._get_obs(), 1, True, {}

        # If tank runs out of gas, they lose
        if (self.black.gas <= 0) and (self.white.gas <= 0):
            return self._get_obs(), 0, True, {}
        elif self.black.gas == 0:  # White tank wins
            return self._get_obs(), -1, True, {}
        elif self.white.gas == 0:  # Black tank wins
            return self._get_obs(), 1, True, {}

        # By default, no rewards are given and game continues
        return self._get_obs(), 0, False, {}

    def _get_obs(self):
        """Compose observation for the agent."""

        # Background is 0
        self.image = np.zeros(self.bf_size, dtype=np.int8)

        # Black tank is 1, white tank is 2
        self.image[self.black.x-self.pad:self.black.x+self.pad+1,
                   self.black.y-self.pad:self.black.y+self.pad+1] = 1
        self.image[self.white.x-self.pad:self.white.x+self.pad+1,
                   self.white.y-self.pad:self.white.y+self.pad+1] = 2

        # Bullets
        for bullet in self.bullets:
            if bullet.dy < 0:    # Bullet moving up is 3
                self.image[bullet.x, bullet.y] = 3
            elif bullet.dx > 0:  # Bullet moving right is 4
                self.image[bullet.x, bullet.y] = 4
            elif bullet.dy > 0:  # Bullet moving down is 5
                self.image[bullet.x, bullet.y] = 5
            elif bullet.dx < 0:  # Bullet moving left is 6
                self.image[bullet.x, bullet.y] = 6

        # Walls
        for wall in self.walls:  # Walls are 7
            self.image[wall.x0:wall.x1, wall.y0:wall.y1] = 7

        # Gather variables from tanks, bullets, walls
        tank_info = [self.black.x,
                     self.black.y,
                     self.black.gas,
                     self.black.cartridge,
                     self.black.HP,
                     self.white.x,
                     self.white.y,
                     self.white.gas,
                     self.white.cartridge,
                     self.white.HP,
                     ]

        max_possible_bullets = 2*(self.bf_side//self.bullet_speed) + 4
        bullet_info = []
        for i in range(max_possible_bullets):
            if i < len(self.bullets):
                bullet = self.bullets[i]
                bullet_info.extend([bullet.x, bullet.y, bullet.dx, bullet.dy])
            else:
                bullet_info.extend([0,0,0,0])

        wall_info = []
        for i, wall in enumerate(self.walls):
            wall_info.extend([wall.x0, wall.y0, wall.x1, wall.y1])

        self.ram = np.array(tank_info + bullet_info + wall_info)
        self.state = (self.image, self.ram)
        return self.state

    def _move_tanks(self, action_b_v, action_w_v):
        """Execute all tank movements."""
        self.black.x += action_b_v.dx
        self.black.y += action_b_v.dy
        self.white.x += action_w_v.dx
        self.white.y += action_w_v.dy

    def _move_bullets(self):
        """Displace all bullets in flight."""
        for i, bullet in enumerate(self.bullets):
            self.bullets[i].x += bullet.dx
            self.bullets[i].y += bullet.dy

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
        # Implemented by keeping bullets inside bounds of battlefield
        self.bullets = list(filter(lambda b: (b.x>=0) and (b.x<self.bf_side),
                                         self.bullets))
        self.bullets = list(filter(lambda b: (b.y>=0) and (b.y<self.bf_side),
                                         self.bullets))

    def _wall_collision(self, tank, action_v):
        """Returns True if the tank would collide with a wall or border if it
        acted on given action.
        """
        dx = action_v.dx
        dy = action_v.dy

        if (dx==0) and (dy==0):  # No need to check this case, can't collide
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

    def _bullet_wall_collisions(self):
        """Remove all bullets that collided in the last step."""
        self.bullets = list(filter(self._bw_no_collision, self.bullets))
        self.new_bullets = list(filter(
            self._new_bw_no_collision, self.new_bullets))

    def _bw_no_collision(self, bullet):
        """Checks collitions between a given bullet and the walls."""
        x = bullet.x
        y = bullet.y
        for i in range(self.bullet_speed):
            if self._point_wall_collision(x, y):
                return False  # This means it did collide
            x -= np.sign(bullet.dx)  # Retract bullet steps this turn
            y -= np.sign(bullet.dy)
        return True

    def _new_bw_no_collision(self, bullet):
        """Checks collitions between a new bullet and the walls."""
        return not self._point_wall_collision(bullet.x, bullet.y)

    def _point_wall_collision(self, x, y):
        """See if a coordinate is part of the walls."""
        try:
            return self.wall_m[x, y] == 1
        except:
            return False  # In case inputed point is out of array bounds

    def _bullet_tank_collision(self, tank):
        """Check if the tank was hit by any bullets."""
        hit = False
        for bullet in self.new_bullets.copy():
            if self._point_in_tank(tank, bullet.x, bullet.y):
                self.new_bullets.remove(bullet)  # Remove the bullet if it hit
                self.hits.append((self.point(bullet.x, bullet.y)))
                hit = True
        for bullet in self.bullets.copy():
            for i in range(self.bullet_speed):  # Look at trail of bullet
                x = bullet.x - i*np.sign(bullet.dx)
                y = bullet.y - i*np.sign(bullet.dy)
                if self._point_in_tank(tank, x, y):
                    self.bullets.remove(bullet)  # Remove the bullet if it hit
                    self.hits.append((self.point(x, y)))
                    hit = True
                    break  # Can't return cause other bullets might hit too
        return hit

    def _point_in_tank(self, tank, x, y):
        """Return true if a point belongs to a tank."""
        return ((x >= tank.x-self.pad) and (x <= tank.x+self.pad) and
                (y >= tank.y-self.pad) and (y <= tank.y+self.pad))

    def _bullet_bullet_collisions(self):
        """Remove all bullets that share same position."""
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

            # Hits
            for hit in self.hits:
                self.render_m[hit.x, hit.y] = "o"

            # Print to console
            for row in self.render_m.tolist():
                print(" ".join(row))
            print(f"Player   Gas   Bullets  HP")
            print(f"Black   {self.black.gas:4}        {self.black.cartridge:3} "
                  f"     {self.black.HP}")
            print(f"White   {self.white.gas:4} {self.white.cartridge:3} "
                  f"{self.white.HP:1}")
            typer.clear()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
