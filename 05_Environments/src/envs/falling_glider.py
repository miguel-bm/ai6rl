"""
Gym Environment that represents a unpowered glider falling from a certain height
and where the only control is the elevator (continous control). Lift and drag
physics are modelled.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class FallingGlider(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.dt = 0.01   # Time step between frames in the environment

        self.g = 9.81    # Acceleration of gravity
        self.rho = 1.25  # Air density

        self.m = 1.      # Mass of the glider
        self.I = 1.      # Moment of inertia along Y axis
        self.Lt = 1.      # Distance between elevator and COM of the glider
        self.Aw = 1.     # Wing surface area
        self.Ae = 0.1    # Elevator surface area

        self.viewer = None

        # Define box for action values
        self.max_eps = 30.   # Maximum elevator angle
        self.min_eps = -30.  # Minimum elevator angle

        # Define box for status values
        low_obs = np.array([-10000.,  # x
                            -10000.,  # y
                            -np.pi,   # theta (angle between centerline and X)
                            -1000.,   # v_x
                            -1000.,   # v_z
                            -1000.,   # omega
                            ], dtype=np.float32)
        high_obs = np.array([10000.,  # x
                             10000.,  # y
                             np.pi,   # theta (angle between centerline and X)
                             1000.,   # v_x
                             1000.,   # v_z
                             1000.,   # omega
                             ], dtype=np.float32)

        self.action_space = spaces.Box(low=self.min_eps, high=self.max_eps,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs,
                                            dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, eps):
        # Make sure action is within bounds
        eps = np.clip(eps, self.min_eps, self.max_eps)
        self.last_eps = eps # for rendering

        # Extract d.o.f. variables from state
        x, z, th, vx, vz, thdot = self.state

        # Calculate full state
        p = np.array([x , 0., z])  # Position vector
        v = np.array([vx, 0., vz])  # Velocity vector
        v_abs = np.linalg.norm(v)  # Absolute velocity
        v_u = (v / v_abs if v_abs != 0.
               else np.array([1., 0., 0.]))  # Velocity versor
        omega = np.array([0., -thdot, 0.])  # Angular velocity vector
        omega_abs = np.linalg.norm(omega)  # Absolute angular velocity
        omega_u = (omega / omega_abs if omega_abs != 0.
                   else np.array([0., 1., 0.]))  # Angular velocity versor
        o_u = np.array([np.cos(th), 0,  np.sin(th)])  # Orientation versor
        v_e = v + np.cross(omega, -o_u * self.Lt)  # Elevator velocity vector
        v_e_abs = np.linalg.norm(v_e)  # Absolute velocity
        v_e_u = (v_e / v_e_abs if v_e_abs != 0.
                else np.array([1., 0., 0.]))  # Elevator velocity versor
        aoa = (th - np.arctan(v_u[2]/v_u[0]) if v_u[0] != 0.
                 else th - np.pi/2)  # Angle Of Attack
        aoa_e = (th + eps - np.arctan(v_e_u[2]/v_e_u[0]) if v_e_u[0] != 0.
                 else th + eps - np.pi/2)  # Elevator A.O.A.

        # Calculate forces over the glider
        Fg = np.array([0, 0, -self.m * self.g])

        Db = 0.5 * self.rho * v_abs**2. * self.Aw * 0.02 * -v_u

        Lw = 0.5 * self.rho * v_abs**2. * self.Aw * lift_coeff(aoa
            ) * np.cross(v_u, np.array([0., 1., 0.]))
        Dw = 0.5 * self.rho * v_abs**2. * self.Aw * drag_coeff(aoa) * -v_u

        Le = 0.5 * self.rho * v_e_abs**2. * self.Ae * lift_coeff(aoa_e
            ) * np.cross(v_e_u, np.array([0., 1., 0.]))
        De = 0.5 * self.rho * v_e_abs**2. * self.Ae * drag_coeff(aoa_e) * -v_e_u

        F = Fg + Db + Lw + Dw + Le + De
        a = F / self.m

        # Calculate moments over the glider
        Mw = 0.
        Me = np.cross(-o_u*self.Lt, Le + De)

        M = Mw + Me
        omegadot = M / self.I

        # Step states
        thdot = thdot - omegadot[1] * self.dt
        vx, _, vz = v + a * self.dt

        th = angle_normalize(th + thdot * self.dt)
        x, _, z = p + v * self.dt

        self.state = np.array([x, z, th, vx, vz, thdot])

        done = True if z <= 0. else False
        reward = vx * self.dt

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = np.array([0.,    # Starts at x=0
                               100.,  # Starts 100 m up
                               -np.pi/2.,  # Starts looking down
                               0.,    # Starts at rest
                               0.,    # Starts at rest
                               0.,    # Starts at rest
                               ], dtype=np.float32)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def render(self, mode='human'):

        # TBD

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def lift_coeff(aoa):
    aoa = angle_normalize(aoa)
    sign = 1 if aoa >= 0 else -1
    aoa = abs(aoa)

    xp = np.linspace(0, np.pi, 30)
    fp = np.array([0.0, 0.4, 0.8, 1.2, 1.6,
                   1.6, 1.3, 1.0, 0.6, 0.4,
                   0.3, 0.2, 0.1, 0.0, 0.0,
                   0.0, -0.1, -0.2, -0.3, -0.4,
                   -0.5, -0.6, -0.7, -0.8, -0.9,
                   -0.8, -0.7, -0.6, -0.3, 0.0,])
    return sign * np.interp(aoa, xp, fp)

def drag_coeff(aoa):
    aoa = angle_normalize(aoa)
    aoa = abs(aoa)

    xp = np.linspace(0, np.pi, 30)
    fp = np.array([0.0  , 0.0029, 0.0117, 0.0263, 0.0467,
                   0.080, 0.1200, 0.1800, 0.2600, 0.4,
                   0.6, 1.0, 1.3, 1.5, 1.6,
                   1.5, 1.3, 1.0, 0.6, 0.4,
                   0.26, 0.18, 0.12, 0.09, 0.07,
                   0.05, 0.03, 0.025, 0.02, 0.015])
    return np.interp(aoa, xp, fp)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
