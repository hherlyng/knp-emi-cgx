import colormaps
import numpy as np
import matplotlib.pyplot as plt

# Plot configuration
figsize = [12, 8]

k = 1 # Finite element degree
dim = 10
N_cells = [100]#, 25, 50, 100, 200]
# N_cells = [5, 10, 25, 50]
timestep = 5e-5
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{dim}m/"
intra_vars = ["Na_i", "K_i", "Cl_i", "phi_i"]
extra_vars = ["Na_e", "K_e", "Cl_e", "phi_e"]

num_timesteps = None
times = None

fig, ax = plt.subplots(figsize=figsize, nrows=len(intra_vars), ncols=2)

for N in N_cells:
    # Read data
    input_dir = output_dir+f"{N}c/"
    ics_point_values = np.load(input_dir+"ics_point_values.npy")
    ecs_point_values = np.load(input_dir+"ecs_point_values.npy")
    if num_timesteps is None:
        num_timesteps = len(ecs_point_values)
        times = np.arange(num_timesteps)*timestep
    # Loop over variables and plot
    for i, var_list in enumerate([intra_vars, extra_vars]):
        if i==0:
            values = ics_point_values
        else:
            values = ecs_point_values

        for j, var in enumerate(var_list):
            print(var, values[-1, j])
            factor = 1e3 if j==len(var_list)-1 else 1        
            ax[j, i].plot(times, factor*values[:, j], label=f"{var}, N={N}")
            ax[j, i].legend()

fig.tight_layout()
fig.savefig(output_dir+f"point_values_{dim}m.png")