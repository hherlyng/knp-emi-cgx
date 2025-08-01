import mpi4py
import pyvista
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx
import numpy as np
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

dim = 5
N_cells = [5, 10, 25, 50, 100]
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{dim}m/"

# Colors
viridis = cm.get_cmap("inferno").resampled(len(N_cells)+1)

dt = 0.00005 # Timestep size in seconds
time_steps = 500 # Number of timesteps
times = np.linspace(0, dt*time_steps, time_steps+1)*1e3 # Times in milliseconds

fig, ax = plt.subplots(figsize=[16, 9])
lw = 2

for i, N in enumerate(N_cells):
    print("N = ", N)
    mesh_version = f"{N}c"
    filename = output_dir+f"{mesh_version}/phi_m.npy"

    phi_m = np.load(filename)
    ax.plot(times, phi_m, label=f"N={N} cells", color=viridis.colors[i], linewidth=lw)

ax.set_ylabel('mV', fontsize=40)
ax.tick_params(axis='both', labelsize=30)
ax.set_xlabel('Time [ms]', fontsize=40) 
ax.legend(fontsize=16, loc='best', frameon=True, fancybox=False, edgecolor='k')
fig.suptitle(f"Membrane potentials cube dimensions = {dim} microns")
fig.tight_layout()
fig.savefig(output_dir+"membrane_potentials")