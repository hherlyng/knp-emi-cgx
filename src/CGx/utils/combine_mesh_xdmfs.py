import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm

mesh_input_filename  = '../KNPEMI/geometries/woudschoten/5m_5c/mesh.xdmf'
facet_input_filename = '../KNPEMI/geometries/woudschoten/5m_5c/facets.xdmf'
mesh_output_filename = '../KNPEMI/geometries/woudschoten/5m_5c/dolfinx_mesh.xdmf'

print(f"Reading mesh with cell tags from {mesh_input_filename}.")
print(f"Reading facet tags from {facet_input_filename}.")
with dfx.io.XDMFFile(comm, mesh_input_filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh")
    mesh.topology.create_entities(mesh.topology.dim-1) # Create facets
    mesh.topology.create_entities(mesh.topology.dim-2) # Create edges
    ct = xdmf.read_meshtags(mesh=mesh, name="mesh") # Read cell tags
    ct.name = "ct"

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

with dfx.io.XDMFFile(comm, facet_input_filename, "r") as xdmf:
    ft = xdmf.read_meshtags(mesh=mesh, name="mesh") # Read facet tags
    ft.name = "ft"

print(f"Number of edges  in mesh: {mesh.topology.index_map(1).size_global}")
print(f"Number of facets in mesh: {mesh.topology.index_map(2).size_global}")
print(f"Number of cells  in mesh: {mesh.topology.index_map(3).size_global}")

print(f"Writing mesh with cell and facet tags to {mesh_output_filename}.")
with dfx.io.XDMFFile(comm, mesh_output_filename, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)