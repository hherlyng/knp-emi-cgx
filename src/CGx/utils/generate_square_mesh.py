import os
import argparse

import dolfinx as dfx

from mpi4py  import MPI
from pathlib import Path
from CGx.utils.misc    import mark_subdomains_square, mark_boundaries_square
from CGx.utils.parsers import CustomParser

description = """
Create a unit square mesh where a sub-square [0.25,0.25]x[0.75,0.75] is tagged as subdomain 1 and the rest as subdomain 2.
The exterior boundary as tagged as 3, the interface between the domains with 4 and all other facets with 5.
"""
parser = argparse.ArgumentParser(formatter_class=CustomParser,
                                 description=description)
parser.add_argument("-N", "--N" ,default=32, type=int, help="Number of elements in each direction")
parser.add_argument("-o", "--output", dest="output_dir", default='./geometries', type=Path, help="Output directory")
args = parser.parse_args()
N = args.N
gm = dfx.mesh.GhostMode.shared_facet
comm = MPI.COMM_WORLD
if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
mesh_filename = args.output_dir / f"square{N}.xdmf"
ft_filename   = args.output_dir / f"square{N}_facets.xdmf"

# Create mesh
mesh = dfx.mesh.create_unit_square(comm, nx=N, ny=N, ghost_mode=gm)

# Generate meshtags
ct = mark_subdomains_square(mesh)
ct.name = "ct"

ft = mark_boundaries_square(mesh)
ft.name = "ft"

with dfx.io.XDMFFile(comm, mesh_filename, 'w') as mesh_xdmf, \
     dfx.io.XDMFFile(comm, ft_filename, 'w') as ft_xdmf:
    mesh_xdmf.write_mesh(mesh)
    mesh_xdmf.write_meshtags(ct, mesh.geometry)
    ft_xdmf.write_mesh(mesh)
    ft_xdmf.write_meshtags(ft, mesh.geometry)
