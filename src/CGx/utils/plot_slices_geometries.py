import mpi4py
import pyvista
import dolfinx
import basix.ufl
import colormaps
import adios4dolfinx
import numpy as np

pyvista.start_xvfb() # Initialize

output_prefix = "/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/"

k = 1 # Finite element degree
intra_vars = ["Na_i", "K_i", "Cl_i", "phi_i"]
extra_vars = ["Na_e", "K_e", "Cl_e", "phi_e"]

EXTRA = 1

# Prepare plotting
zoom = 1.5

# Colors
cmap = colormaps.matter

# Scalar bar args
sargs = {
    'n_labels': 10, 
    'fmt': '%.2g',
    'font_family': 'arial',
    'title_font_size' : 20,
    'label_font_size' : 20,
    'position_x' : 0.075,
    'position_y' : 0.10,
    'vertical' : True,
    'height' : 0.75,
}

times = np.arange(501)
time = times[100]

# Prepare finite elements and pyvista grid used to plot
dims = [5, 10, 20]
N_cells = [10, 25, 50, 100]
pl = pyvista.Plotter(shape=(len(N_cells), len(dims)),
                    window_size=[2000, 2000],
                    border=False,
                    off_screen=True
                )

for i, dim in enumerate(dims):
    print(f"i = {i}")

    # Slice coordinates
    approx_bounds = dim*1e3*1e-9
    origin_yz = [approx_bounds/2, approx_bounds/2, approx_bounds/2]
        
    for j, N in enumerate(N_cells):
        print(f"j = {j}")
        filename = output_prefix+f"{dim}m/{N}c/checkpoints" 
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

        # Read concentration
        adios4dolfinx.read_function(filename, u_cg, time=time, name=f"K_e")
        
        # Set data
        grid.point_data["c"] = u_cg.x.array.copy()

        # Use the threshold filter to select cells that are in the extracellular space
        extra_grid = grid.threshold([EXTRA, EXTRA], scalars="Domain ID")
        sliced_grid = extra_grid.clip(normal=[-1, 0, 0], origin=origin_yz, invert=False)
        if dim==5:
            sliced_grid = sliced_grid.threshold(10, invert=True)
            
        sargs['title'] = f'c_K_e [mM], {N} cells, {dim} mu'
        # Plot yz plane
        pl.subplot(j, i)
        pl.add_mesh(sliced_grid,
                    cmap=cmap,
                    scalar_bar_args=sargs.copy()
                    )
        pl.view_isometric()
        pl.camera.zoom(zoom)

        pl.enable_parallel_projection()
        pl.background_color = 'white'

pl.screenshot(output_prefix+"slices_varying_geometry.png")