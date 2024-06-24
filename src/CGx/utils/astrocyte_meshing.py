import dolfinx as dfx

from mpi4py import MPI

# Set mesh parameters and filename
gm = dfx.mesh.GhostMode.shared_facet
comm = MPI.COMM_WORLD
filename1 = '../../EMI/data/astrocyte_mesh_physical_region.xdmf'
filename2 = '../../EMI/data/astrocyte_mesh_facet_region.xdmf'
filename3 = '../../EMI/data/astrocyte_mesh_full.xdmf'


with dfx.io.XDMFFile(comm, filename1, 'r') as xdmf:
    mesh = xdmf.read_mesh(name="mesh")
    ct   = xdmf.read_meshtags(mesh, name="mesh")
    ct.name = "ct"
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

with dfx.io.XDMFFile(comm, filename2, 'r') as xdmf:
    ft    = xdmf.read_meshtags(mesh, name="mesh")
    ft.name = "ft"
from IPython import embed;embed()
with dfx.io.XDMFFile(comm, filename3, 'w') as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, x=mesh.geometry)
    xdmf.write_meshtags(ft, x=mesh.geometry)