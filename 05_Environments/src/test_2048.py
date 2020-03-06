from envs.g2048 import g2048
import time
import numpy as np

env = g2048()
obs = env.reset()
for i in range(10):
    time.sleep(1.)
    env.render()
    obs, reward, done, _ = env.step(np.random.choice([0, 1, 2, 3]))


#for i in range(10):
#    obs, reward, done, _ = env.step(0)
