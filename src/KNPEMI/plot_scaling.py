import matplotlib.pyplot as plt
import numpy as np


t_s = np.array([490, 403, 389, 257, 293, 260]) # solve time in seconds
t_a = np.array([104, 51.7, 36, 27.6, 23.5, 20.6]) # assembly time in seconds
cores = np.arange(1, 7)

fig, ax = plt.subplots()
ax.semilogy(cores, t_s, label='solve')
ax.semilogy(cores, t_a, label='assembly')
ax.legend()
plt.show()