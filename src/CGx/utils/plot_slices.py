import mpi4py
import pyvista
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx
import numpy as np
from matplotlib import colormaps as cm

k = 1 # Finite element degree
dim = 5
N_cells = 50
mesh_version = f"{dim}m/{N_cells}c"
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{mesh_version}/"
filename = output_dir+"checkpoints" 
intra_vars = ["Na_i", "K_i", "Cl_i", "phi_i"]
extra_vars = ["Na_e", "K_e", "Cl_e", "phi_e"]

EXTRA = 1

# Prepare plotting
# Slice coordinates
approx_bounds = dim*1e3*1e-9
origin_yz = [approx_bounds/2, approx_bounds/2, approx_bounds/2]
# origin_xz = [0.0, -0.0112, 0.0075]
# origin_xy = [0.0, 0.0029, 0.0065]

zoom_yz = 1.5
# zoom_xz = 1.0
# zoom_xy = 1.35

# Colors
viridis = cm.get_cmap("viridis")
inferno = cm.get_cmap("inferno")
cmaps = {0 : viridis,
         1 : inferno
}
n_colors = 16
sargs = sargs = {
    'title': 'Concentration [mM]',
    'n_labels': 4, 
    'fmt': '%.2g',
    'font_family': 'arial'
}

pl = pyvista.Plotter(shape=(4, 2), window_size=[1000, 1200], border=False)

view = 'yz'

times = np.arange(501)*5e-5
time = times[2]

# Prepare finite elements and pyvista grid used to plot
mesh = adios4dolfinx.read_mesh(filename, comm=mpi4py.MPI.COMM_WORLD)
ct   = adios4dolfinx.read_meshtags(filename, mesh, meshtag_name="ct")
CG = dolfinx.fem.functionspace(mesh,
        basix.ufl.element("CG", mesh.basix_cell(), k)
        )
u_cg = dolfinx.fem.Function(CG)

# Get the intracellular tags
INTRA = np.unique(ct.values)[1:]
intra_cells_idx = np.concatenate(([ct.find(tag) for tag in INTRA]))
extra_cells_idx = ct.find(EXTRA)

cells, types, x = dolfinx.plot.vtk_mesh(CG)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.cell_data["Domain ID"] = ct.values

for i, var_list in enumerate([intra_vars, extra_vars]):
        
    for j, var in enumerate(var_list):
        # Read concentration
        adios4dolfinx.read_function(filename, u_cg, time=float(time), name=f"{var}")
        
        # Set data
        grid.point_data["c"] = u_cg.x.array.copy()

        # Use the threshold filter to select cells that are in the intracellular
        # and extracellular spaces
        intra_grid = grid.threshold([INTRA[0], INTRA[-1]], scalars="Domain ID")
        extra_grid = grid.threshold([EXTRA, EXTRA], scalars="Domain ID")

        if i==0:
            sliced_grid = intra_grid.clip(normal=[-1, 0, 0], origin=origin_yz, invert=False)
        else:
            sliced_grid = extra_grid.clip(normal=[-1, 0, 0], origin=origin_yz, invert=False)
        
        sargs['title'] = f'Time: {time}, var: {var}' # Give each bar a unique title
        
        if view=='yz':
            # Plot yz plane
            pl.subplot(j, i)
            pl.add_mesh(sliced_grid,
                        cmap=cmaps[i],
                        # clim=[sliced_grid, 0.75] if location=="laterals" else [0, 1.0],
                        show_scalar_bar=True,
                        scalar_bar_args=sargs.copy(),
                        n_colors=n_colors)
            # pl.add_scalar_bar(scalar_bar_text)
            # pl.view_yz()
            pl.view_isometric()
            pl.camera.zoom(zoom_yz)

        pl.enable_parallel_projection()
        pl.background_color = 'white'

pl.show(screenshot=output_dir+f"slices.png")