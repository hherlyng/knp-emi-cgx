import mpi4py
import pyvista
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx
import numpy as np
from matplotlib import colormaps as cm

k = 1 # Finite element degree
dim = 10
N_cells = 100
mesh_version = f"{dim}m/{N_cells}c"
output_dir = f"/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/{mesh_version}/"
filename = output_dir+"checkpoints" 

EXTRA = 1

# Prepare plotting
# Slice coordinates
approx_bounds = dim*1e3*1e-9
origin_yz = [approx_bounds/2, approx_bounds/2, approx_bounds/2]
# origin_xz = [0.0, -0.0112, 0.0075]
# origin_xy = [0.0, 0.0029, 0.0065]

zoom_yz = 1.4
# zoom_xz = 1.0
# zoom_xy = 1.35

# Colors
cmap = colormaps.curl
n_colors = 64
sargs = sargs = {
    'title': 'Membrane potential [mV]',
    'n_labels': 4, 
    'fmt': '%.2g',
    'font_family': 'arial'
}

pl = pyvista.Plotter(shape=(3, 1), window_size=[500, 1200], border=False) 

view = 'yz'

dt = 5e-5
num_timesteps = 400
times = np.arange(num_timesteps)
timestamps = [times[5], times[50], times[100]]

# Prepare finite elements and pyvista grid used to plot
mesh = adios4dolfinx.read_mesh(filename, comm=mpi4py.MPI.COMM_WORLD)
ct   = adios4dolfinx.read_meshtags(filename, mesh, meshtag_name="ct")
CG = dolfinx.fem.functionspace(mesh,
        basix.ufl.element("CG", mesh.basix_cell(), k)
        )
phi_m = dolfinx.fem.Function(CG)

# Get the intracellular tags
INTRA = np.unique(ct.values)[1:]
intra_cells_idx = np.concatenate(([ct.find(tag) for tag in INTRA]))
extra_cells_idx = ct.find(EXTRA)

cells, types, x = dolfinx.plot.vtk_mesh(CG)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.cell_data["Domain ID"] = ct.values

for i, timestamp in enumerate(timestamps):
    adios4dolfinx.read_function(filename, phi_m, time=timestamp, name=f"phi_m")

    # Set data
    grid.point_data["phi_m"] = phi_m.x.array.copy()*1e3 # Convert to mV

    # Use the threshold filter to select cells that are in the intracellular
    # and extracellular spaces
    intra_grid = grid.threshold([55, 70], scalars="Domain ID")
    sliced_grid = intra_grid.clip(normal=[-1, 0, 0], origin=origin_yz, invert=False)
        
    sargs['title'] = f'Time: {timestamp*dt*1e3} ms, Membrane potential [mV]' # Give each bar a unique title
            
    if view=='yz':
        # Plot yz plane
        pl.subplot(i, 0)
        pl.add_mesh(sliced_grid,
                    cmap=cmap,
                    # clim=[sliced_grid, 0.75] if location=="laterals" else [0, 1.0],
                    show_scalar_bar=True,
                    scalar_bar_args=sargs.copy(),
                    n_colors=n_colors)
        # pl.add_scalar_bar(scalar_bar_text)
        pl.view_yz()
        pl.view_isometric()
        pl.camera.zoom(zoom_yz)
        pl.camera.azimuth = -45
        pl.camera.elevation = -30

# pl.enable_parallel_projection()
pl.background_color = 'white'

pl.show(screenshot=output_dir+f"phi_m_slice.png")