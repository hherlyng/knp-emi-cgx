from mpi4py import MPI
import sys
import numpy as np
import dolfinx as dfx

# Mesh directory and filenames
mesh_dir = sys.argv[1]
mesh_file = sys.argv[2]
facets_file = sys.argv[3]

if mesh_file==facets_file:
    ct_name = "ct"
    ft_name = "ft"
else:
    ct_name = ft_name = "mesh"

# Load mesh and cell tags
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_dir + mesh_file, "r") as xdmf:
    mesh = xdmf.read_mesh()
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim-1
    mesh.topology.create_entities(tdim-2)
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    ct = xdmf.read_meshtags(mesh, ct_name)
    if mesh_file==facets_file:
        ft = xdmf.read_meshtags(mesh, ft_name)

# Load facet tags
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_dir + facets_file, "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, ft_name)

# Refine mesh
mesh_refined, parent_cells, parent_facets = \
    dfx.mesh.refine(mesh,
                option=dfx.mesh.RefinementOption.parent_cell_and_facet
        )
mesh_refined.topology.create_connectivity(fdim, tdim)

# Get child and parent vertices
child_vertices = dfx.mesh.entities_to_geometry(
                        mesh_refined,
                        tdim,
                        np.arange(len(parent_cells), dtype=np.int32)
                    )
parent_vertices = dfx.mesh.entities_to_geometry(mesh, tdim, parent_cells)

# Transfer meshtags to the refined mesh
ft_refined = dfx.mesh.MeshTags(
                dfx.cpp.refinement.transfer_facet_meshtag(
                    ft._cpp_object,
                    mesh_refined._cpp_object.topology,
                    parent_cells,
                    parent_facets
                )
            )
ct_refined = dfx.mesh.MeshTags(
                dfx.cpp.refinement.transfer_cell_meshtag(
                    ct._cpp_object,
                    mesh_refined._cpp_object.topology,
                    parent_cells
                )
            )
# Rename meshtags
ft_refined.name = "ft"
ct_refined.name = "ct"

with dfx.io.XDMFFile(mesh_refined.comm, mesh_dir + mesh_file + "_refined.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_refined)
    xdmf.write_meshtags(ct_refined, mesh_refined.geometry)
    xdmf.write_meshtags(ft_refined, mesh_refined.geometry)