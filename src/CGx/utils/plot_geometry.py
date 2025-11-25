import pyvista
import colormaps
import numpy as np

pyvista.start_xvfb() # Initialize

output_filename = "/global/D1/homes/hherlyng/knp-emi-cgx/output/GC/single_geometry_edges.png"
input_prefix  = "/home/hherlyng/knp-emi-cgx/src/CGx/KNPEMI/input/geometries/GC/"
EXTRA = 1

# Prepare plotting
zoom = 1.19

# Colors
cmap = colormaps.delta
sargs = {
    'fmt': '%.0f',
    'font_family': 'arial',
    'title_font_size' : 60,
    'label_font_size' : 60,
    'position_x' : 0.125,
    'position_y' : 0.01,
    'width' : 0.75,
    'n_labels' : 0,
    'title' : "Cell label"
}

# Prepare finite elements and pyvista grid used to plot
dim = 20
N = 500
pl = pyvista.Plotter(window_size=[2400, 2400], border=False, off_screen=True)

filename = input_prefix+f"{dim}m/{N}c/mesh.xdmf" 
mesh = pyvista.read_meshio(filename)

sargs['n_colors'] = N
clim = [2, N]

# Plot yz plane
pl.add_mesh(mesh.threshold(1.5), # Remove the extracellular space
            cmap=cmap,
            clim=clim,
            # scalar_bar_args=sargs.copy(),
            show_scalar_bar=False,
            show_edges=True,
            edge_color='white',
            edge_opacity=0.35
            )
# pl.scalar_bar.GetTitleTextProperty().SetLineSpacing(2)
pl.camera.azimuth = 180
pl.camera.elevation = -20
pl.camera.zoom(zoom)
pos = pl.camera.position
foc = pl.camera.focal_point
pl.camera.position = (pos[0], pos[1], pos[2] - 3500)
pl.camera.focal_point = (foc[0], foc[1], foc[2] - 3500)

pl.screenshot(output_filename)