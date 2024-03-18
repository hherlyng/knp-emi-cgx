import sys

import dolfinx as dfx

from utils_dfx import mark_subdomains_square, mark_boundaries_square
from mpi4py import MPI

flag = 'w' # Read or write

# Set mesh parameters and filename
N = int(sys.argv[1])
gm = dfx.mesh.GhostMode.shared_facet
comm = MPI.COMM_WORLD
filename = './geometries/square' + str(N) + '.xdmf'


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