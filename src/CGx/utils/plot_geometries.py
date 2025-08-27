import pyvista
import colormaps
import numpy as np

pyvista.start_xvfb() # Initialize

output_filename = "/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/geometries.png"
input_prefix  = "/home/hherlyng/knp-emi-cgx/src/CGx/KNPEMI/geometries/GC/"
EXTRA = 1

# Prepare plotting
zoom = 1.25

# Colors
cmap = colormaps.delta
sargs = {
    'fmt': '%.0f',
    'font_family': 'arial',
    'title_font_size' : 40,
    'position_x' : 0.075,
    'position_y' : 0.10,
    'vertical' : True,
    'height' : 0.75,
    'n_labels' : 0
}


times = np.arange(501)
time = times[200]

# Prepare finite elements and pyvista grid used to plot
dims = [5, 10, 20]
N_cells = [100, 200, 300, 400, 500]
pl = pyvista.Plotter(shape=(len(N_cells), len(dims)), window_size=[1800, 2000], border=False, off_screen=True, image_scale=2)

for i, dim in enumerate(dims):
    print(f"i = {i}")
        
    for j, N in enumerate(N_cells):
        print(f"j = {j}")
        filename = input_prefix+f"{dim}m/{N}c/mesh.xdmf" 
        mesh = pyvista.read_meshio(filename)
        
        sargs['n_colors'] = N
        sargs['title'] = f"{N} cells"
        clim = [2, mesh.cell_data["label"].max()]
        print("Number of unique cells: ", clim[1])
        
        # Plot yz plane
        pl.subplot(j, i)
        pl.add_mesh(mesh.threshold(1.5), # Remove the extracellular space
                    cmap=cmap,
                    clim=clim,
                    scalar_bar_args=sargs.copy(),
                    show_scalar_bar=False,
                    show_edges=True,
                    edge_color="white",
                    edge_opacity=0.25
                    )
        # pl.scalar_bar.GetTitleTextProperty().SetLineSpacing(3)
        pl.camera.azimuth = 180
        pl.camera.elevation = -20
        pl.camera.zoom(zoom)

pl.screenshot(output_filename)