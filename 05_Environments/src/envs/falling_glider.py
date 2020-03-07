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
        self.max_eps = 0.5   # Maximum elevator angle (rad)
        self.min_eps = -0.5  # Minimum elevator angle (rad)

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

        self.last_eps = None

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

        v_ang = np.arccos(v_u[0]) * (np.sign(v_u[2]) if v_u[2] != 0 else 1.)
        aoa = th - v_ang  # Angle Of Attack

        v_e_ang = np.arccos(v_e_u[0]) * (np.sign(v_e_u[2])
            if v_e_u[2] != 0 else 1.)
        aoa_e = th + eps - v_e_ang  # Elevator A.O.A.

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
        self.last_eps = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def render(self, mode='human'):

        # TBD

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1250,600)
            self.viewer.set_bounds(-50., 200., -10.,110.)

            glider_shape = [(-10, -1), (4, -1), (5, -0.5), (5, 0),
                            (1, 1), (-2, 1), (-3, 0.5), (-4, -0.5),
                            (-8.5, -0.5), (-10, 1), (-10.5, 1)]

            self.glider = rendering.make_polygon(glider_shape, filled=True)
            self.glider.set_color(0, 0, 0)
            self.glidertrans = rendering.Transform()
            self.glider.add_attr(self.glidertrans)

            wing_shape = [(-3, -0.25), (0, -0.5), (1, -0.4), (2, -0.25),
                          (1, -0.1), (0, 0)]

            self.wing = rendering.make_polygon(wing_shape, filled=True)
            self.wing.set_color(255, 0, 0)
            self.wingtrans = rendering.Transform()
            self.wing.add_attr(self.wingtrans)

            elev_shape = [(-1.5, -0.25), (0, -0.5), (0.5, -0.4), (1, -0.25),
                          (0.5, -0.1), (0, 0)]

            self.elev = rendering.make_polygon(elev_shape, filled=True)
            self.elev.set_color(255, 0, 255)
            self.elevtrans = rendering.Transform()
            self.elev.add_attr(self.elevtrans)

        # Draw the sea
        self.viewer.draw_polygon([(-50, -50), (200, -50), (200, 0), (-50, 0)],
            color=(0, 0, 255))



        self.viewer.add_onetime(self.glider)
        self.glidertrans.translation = (self.state[0], self.state[1])
        self.glidertrans.rotation = (self.state[2])

        self.viewer.add_onetime(self.wing)
        self.wingtrans.translation = (self.state[0], self.state[1])
        self.wingtrans.rotation = (self.state[2])

        if self.last_eps is not None:
            x0, y0, th = self.state[0], self.state[1], self.state[2]
            Lt = 9.6
            x = x0 - Lt * np.cos(th)
            y = y0 - Lt * np.sin(th)
            self.viewer.add_onetime(self.elev)
            self.elevtrans.translation = (x, y)
            self.elevtrans.rotation = (th + self.last_eps)


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
