from envs.tank6 import TankSaturdays
import matplotlib.pyplot as plt
import time
import numpy as np

env = TankSaturdays()
obs = env.reset()


for i in range(500):
    env.render()
    obs, reward, done, _ = env.step(np.random.randint(9))

    time.sleep(0.2)
