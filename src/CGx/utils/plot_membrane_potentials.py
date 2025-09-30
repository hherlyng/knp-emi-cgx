import sys
import mpi4py
import pyvista
import dolfinx
import basix.ufl
import adios4dolfinx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

dim = int(sys.argv[1])
N_cells = int(sys.argv[2])
mesh_version = f"{dim}m/{N_cells}c"
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{mesh_version}/"
filename = output_dir+"checkpoints" 

k = 1 # Finite element degree

timestep = 5e-5
num_timesteps = int(sys.argv[3])
T = (num_timesteps)*timestep
times_int = np.arange(num_timesteps+1)

# Prepare finite elements and pyvista grid used to plot
mesh = adios4dolfinx.read_mesh(filename, comm=mpi4py.MPI.COMM_WORLD)
ct   = adios4dolfinx.read_meshtags(filename, mesh, meshtag_name="ct")
ft   = adios4dolfinx.read_meshtags(filename, mesh, meshtag_name="ft")
CG = dolfinx.fem.functionspace(mesh,
        basix.ufl.element("CG", mesh.basix_cell(), k)
        )
u_cg = dolfinx.fem.Function(CG)

cells, types, x = dolfinx.plot.vtk_mesh(CG)
grid = pyvista.UnstructuredGrid(cells, types, x)

xx, yy, zz = [mesh.geometry.x[:, i] for i in range(mesh.geometry.dim)]
x_min = xx.min()
x_max = xx.max()
y_min = yy.min()
y_max = yy.max()
z_min = zz.min()
z_max = zz.max()
x_c = (x_max + x_min) / 2
y_c = (y_max + y_min) / 2
z_c = (z_max + z_min) / 2
mesh_center = np.array([x_c, y_c, z_c])

cell_ids = [66, 68] # 1st stimulated, 2nd neighbor

gamma_facets = ft.find(cell_ids[0])
gamma_vertices = dolfinx.mesh.compute_incident_entities(
                                                mesh.topology,
                                                gamma_facets,
                                                mesh.topology.dim-1,
                                                0
)
gamma_vertices = np.unique(gamma_vertices)
gamma_coords = mesh.geometry.x[gamma_vertices]

# Find the vertex that lies closest to the cell's centroid
distances = np.sum((gamma_coords - mesh_center)**2, axis=1)
argmin_local = np.argmin(distances)
min_dist_local = distances[argmin_local]
min_vertex = gamma_vertices[argmin_local]

min_point1 = mesh.geometry.x[min_vertex]

# Find the cell that contains the vertex
bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, min_point1)
colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, min_point1)
cc = colliding_cells.links(0)[0]
membrane_cell1 = np.array(cc)

gamma_facets = ft.find(cell_ids[1]) 
gamma_vertices = dolfinx.mesh.compute_incident_entities(
                                                mesh.topology,
                                                gamma_facets,
                                                mesh.topology.dim-1,
                                                0
)
gamma_vertices = np.unique(gamma_vertices)
gamma_coords = mesh.geometry.x[gamma_vertices]

# Find the vertex that lies closest to the cell's centroid
distances = np.sum((gamma_coords - mesh_center)**2, axis=1)
argmin_local = np.argmin(distances)
min_dist_local = distances[argmin_local]
min_vertex = gamma_vertices[argmin_local]

min_point2 = mesh.geometry.x[min_vertex]

# Find the cell that contains the vertex
bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, min_point2)
colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, min_point2)
cc = colliding_cells.links(0)[0]
membrane_cell2 = np.array(cc)

phi_m_1 = []
phi_m_2 = []

n_1 = []
m_1 = []
h_1 = []

n_2 = []
m_2 = []
h_2 = []

for t in times_int:
    # Read potentials
    adios4dolfinx.read_function(filename, u_cg, time=t, name=f"phi_m")
    phi_m_1.append(u_cg.eval(min_point1, membrane_cell1))
    phi_m_2.append(u_cg.eval(min_point2, membrane_cell2))

    # Read gating variables
    adios4dolfinx.read_function(filename, u_cg, time=t, name=f"n")
    n_1.append(u_cg.eval(min_point1, membrane_cell1))
    n_2.append(u_cg.eval(min_point2, membrane_cell2))

    adios4dolfinx.read_function(filename, u_cg, time=t, name=f"m")
    m_1.append(u_cg.eval(min_point1, membrane_cell1))
    m_2.append(u_cg.eval(min_point2, membrane_cell2))
    
    adios4dolfinx.read_function(filename, u_cg, time=t, name=f"h")
    h_1.append(u_cg.eval(min_point1, membrane_cell1))
    h_2.append(u_cg.eval(min_point2, membrane_cell2))

# Plot membrane potentials
figsize = [12, 8]
fig, ax = plt.subplots(figsize=figsize, ncols=2)

# Plot in milliseconds and millivolts
ax[0].plot(times_int*timestep*1e3, np.array(phi_m_1)*1e3)
ax[1].plot(times_int*timestep*1e3, np.array(phi_m_2)*1e3)

fig.tight_layout()
fig.savefig(output_dir+f"membrane_potentials_stim_and_neighbor.png")

# Plot gating variables
fig, ax = plt.subplots(figsize=figsize, ncols=2)

# Plot in milliseconds
ax[0].plot(times_int*timestep*1e3, n_1, label=f"n")
ax[0].plot(times_int*timestep*1e3, m_1, label=f"m")
ax[0].plot(times_int*timestep*1e3, h_1, label=f"h")
ax[0].legend()
ax[1].plot(times_int*timestep*1e3, n_2, label=f"n")
ax[1].plot(times_int*timestep*1e3, m_2, label=f"m")
ax[1].plot(times_int*timestep*1e3, h_2, label=f"h")
ax[1].legend()

fig.tight_layout()
fig.savefig(output_dir+f"gating_variables_stim_and_neighbor.png")