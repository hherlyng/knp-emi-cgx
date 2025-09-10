import colormaps
import numpy as np
import matplotlib.pyplot as plt

# Plot configuration
figsize = [12, 8]

k = 1 # Finite element degree
timestep = 0.0000125
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/"
input_dir = output_dir
# intra_vars = ["Na_i", "K_i", "phi_i"]
# extra_vars = ["Na_e", "K_e", "phi_e"]
intra_vars = ["Na_i", "K_i", "Cl_i", "phi_i"]
extra_vars = ["Na_e", "K_e", "Cl_e", "phi_e"]

num_timesteps = None
times = None

fig, ax = plt.subplots(figsize=figsize, nrows=len(intra_vars), ncols=2)

# Read data
ics_point_values = np.load(input_dir+"ics_point_values.npy")
ecs_point_values = np.load(input_dir+"ecs_point_values.npy")
if num_timesteps is None:
    num_timesteps = len(ecs_point_values)
    times = np.arange(num_timesteps)*timestep*1e3 # Times in milliseconds
# Loop over variables and plot
for i, var_list in enumerate([intra_vars, extra_vars]):
    if i==0:
        values = ics_point_values
    else:
        values = ecs_point_values

    for j, var in enumerate(var_list):
        print(var, values[-1, j])
        factor = 1e3 if j==len(var_list)-1 else 1        
        ax[j, i].plot(times, factor*values[:, j], label=f"{var}")
        ax[j, i].legend()

fig.tight_layout()
fig.savefig(output_dir+f"point_values_square.png")