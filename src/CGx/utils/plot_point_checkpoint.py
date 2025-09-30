import mpi4py
import dolfinx
import basix.ufl
import adios4dolfinx
import numpy as np
import matplotlib.pyplot as plt
import scifem

k = 1 # Finite element degree
dim = 10
N_cells = 100
timestep = 5e-5
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{dim}m/{N_cells}c/"
filename = output_dir + "checkpoints"
intra_vars = ["Na_i", "K_i", "Cl_i"]
extra_vars = ["Na_e", "K_e", "Cl_e"]

num_timesteps = 5 #400
times = np.arange(num_timesteps)

# Prepare finite elements and pyvista grid used to plot
mesh = adios4dolfinx.read_mesh(filename, comm=mpi4py.MPI.COMM_WORLD)
ct   = adios4dolfinx.read_meshtags(filename, mesh, meshtag_name="ct")
CG = dolfinx.fem.functionspace(mesh,
        basix.ufl.element("CG", mesh.basix_cell(), k)
        )
u = dolfinx.fem.Function(CG)

# Plot configuration
figsize = [12, 8]
fig, ax = plt.subplots(figsize=figsize, nrows=len(intra_vars), ncols=2)

point = np.array([100, 100, 100])
u_values = []

# Loop over variables and plot
for i, var_list in enumerate([intra_vars, extra_vars]):
    for j, var in enumerate(var_list):
        for time in times:
            # Read concentration
            adios4dolfinx.read_function(filename, u, time=time, name=f"{var}")
            u_values.append(evaluate_function(u, point))
                  
        ax[j, i].plot(times, u_values, label=f"{var}")
        ax[j, i].legend()

        u_values = [] # Zero out values

fig.tight_layout()
fig.savefig(output_dir+f"point_values_{dim}m_from_checkpoint.png")