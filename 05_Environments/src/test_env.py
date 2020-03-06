from envs.falling_glider import FallingGlider
import matplotlib.pyplot as plt
import time
import numpy as np

env = FallingGlider()
obs = env.reset()

x = []
z = []
for i in range(3000):
    env.render()
    obs, reward, done, _ = env.step(np.random.uniform(-0.05, -0.04))

    x.append(obs[0])
    z.append(obs[1])
    time.sleep(0.01)


plt.plot(x, z)
plt.savefig("a.png")
