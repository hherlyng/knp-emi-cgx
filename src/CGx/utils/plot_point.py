import mpi4py
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx
import numpy as np
import matplotlib.pyplot as plt

# Plot configuration
cmap = colormaps.matter
figsize = [12, 8]

k = 1 # Finite element degree
dim = 5
N_cells = [5, 10, 25]
num_timesteps = 500
timestep = 5e-5
output_dir = "/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/"
intra_vars = ["Na_i", "K_i", "Cl_i", "phi_i"]
extra_vars = ["Na_e", "K_e", "Cl_e", "phi_e"]

times = np.arange(num_timesteps+1)*timestep

fig, ax = plt.subplots(figsize=figsize, nrows=len(intra_vars), ncols=2)

for N in N_cells:
    # Read data
    input_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{dim}m/{N}c/"
    ics_point_values = np.load(input_dir+"ics_point_values.npy")
    ecs_point_values = np.load(input_dir+"ecs_point_values.npy")

    # Loop over variables and plot
    for i, var_list in enumerate([intra_vars, extra_vars]):
        if i==0:
            values = ics_point_values
        else:
            values = ecs_point_values

        for j, var in enumerate(var_list):
            factor = 1e3 if j==len(var_list)-1 else 1        
            ax[j, i].plot(times, factor*values[:, j], label=f"{var}, N={N}")
            ax[j, i].legend()

fig.tight_layout()
fig.savefig(output_dir+"point_values.png")