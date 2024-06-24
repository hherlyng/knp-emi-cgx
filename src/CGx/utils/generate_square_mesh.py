import argparse
from pathlib import Path
from mpi4py import MPI
import dolfinx as dfx

from CGx.utils.misc import mark_subdomains_square, mark_boundaries_square
from CGx.KNPEMI.parsers import CustomParser
description = """
Create a unit square mesh where a sub-square [0.25,0.25]x[0.75,0.75] is tagged as subdomain 1 and the rest as subdomain 2.
The exterior boundary as tagged as 3, the interface between the domains with 4 and all other facets with 5.
"""
parser = argparse.ArgumentParser(formatter_class=CustomParser,
                                 description=description)
parser.add_argument("-N", "--N" ,default=20, type=int, help="Number of elements in each direction")
parser.add_argument("-f", "--flag", dest="flag", default='w', type=str, choices=["r", "w"], help="Read or write")
parser.add_argument("-o", "--output", dest="output_dir", default='./geometries', type=Path, help="Output directory")
args = parser.parse_args()
flag = args.flag
N = args.N
gm = dfx.mesh.GhostMode.shared_facet
comm = MPI.COMM_WORLD
filename = args.output_dir / f"square{N}.xdmf"


# Create mesh
mesh = dfx.mesh.create_unit_square(comm, nx=N, ny=N, ghost_mode=gm)

# Generate meshtags
ct = mark_subdomains_square(mesh)
ct.name = "ct"

ft = mark_boundaries_square(mesh)
ft.name = "ft"

if flag == 'w':
    with dfx.io.XDMFFile(comm, filename, flag) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, x=mesh.geometry)
        xdmf.write_meshtags(ft, x=mesh.geometry)

elif flag == 'r':
    with dfx.io.XDMFFile(comm, filename, flag) as xdmf:
        mesh = xdmf.read_mesh(name = "mesh")
        ct   = xdmf.read_meshtags(mesh, name = "ct")
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        ft = xdmf.read_meshtags(mesh, name = "ft")