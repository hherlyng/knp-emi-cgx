import os
import ufl
import warnings
import numpy.typing

import numpy   as np
import dolfinx as dfx

from scipy    import sparse
from petsc4py import PETSc

# Utility functions
def norm_2(vec):
    return ufl.sqrt(ufl.dot(vec, vec))

def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)
            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])

def flatten_list(input_list):
    return [item for sublist in input_list for item in (sublist if isinstance(sublist, tuple) else [sublist])]

def check_if_file_exists(file_path):

    if not os.path.exists(file_path):        
        print(f"The file '{file_path}' does not exist.")
        exit()

def compute_interface_integration_entities(
        msh, interface_facets, domain_0_cells, domain_1_cells,
        domain_to_domain_0, domain_to_domain_1):
    """
    
    Copyright (c) 2024 Joseph P. Dean. This function is written by Joe Dean,
    subject to copyright under an MIT License.

    This function computes the integration entities (as a list of pairs of
    (cell, local facet index) pairs) required to assemble mixed domain forms
    over the interface. It assumes there is a domain with two sub-domains,
    domain_0 and domain_1, that have a common interface. Cells in domain_0
    correspond to the "+" restriction and cells in domain_1 correspond to
    the "-" restriction.

    Parameters:
        interface_facets: A list of facets on the interface
        domain_0_cells: A list of cells in domain_0
        domain_1_cells: A list of cells in domain_1
        c_to_f: The cell to facet connectivity for the domain mesh
        f_to_c: the facet to cell connectivity for the domain mesh
        facet_imap: The facet index_map for the domain mesh
        domain_to_domain_0: A map from cells in domain to cells in domain_0
        domain_to_domain_1: A map from cells in domain to cells in domain_1

    Returns:
        interface_entities: The integration entities
        domain_to_domain_0: A modified map (see HACK below)
        domain_to_domain_1: A modified map (see HACK below)
    """
    # Create measure for integration. Assign the first (cell, local facet)
    # pair to the cell in domain_0, corresponding to the "+" restriction.
    # Assign the second pair to the domain_1 cell, corresponding to the "-"
    # restriction.
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # FIXME This can be done more efficiently
    interface_entities = []
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            cell_plus = cells[0] if cells[0] in domain_0_cells else cells[1]
            cell_minus = cells[0] if cells[0] in domain_1_cells else cells[1]
            assert cell_plus in domain_0_cells
            assert cell_minus in domain_1_cells

            # FIXME Don't use tolist
            local_facet_plus = c_to_f.links(
                cell_plus).tolist().index(facet)
            local_facet_minus = c_to_f.links(
                cell_minus).tolist().index(facet)
            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus])

            # FIXME HACK cell_minus does not exist in the left submesh, so it
            # will be mapped to index -1. This is problematic for the
            # assembler, which assumes it is possible to get the full macro
            # dofmap for the trial and test functions, despite the restriction
            # meaning we don't need the non-existant dofs. To fix this, we just
            # map cell_minus to the cell corresponding to cell plus. This will
            # just add zeros to the assembled system, since there are no
            # u("-") terms. Could map this to any cell in the submesh, but
            # I think using the cell on the other side of the facet means a
            # facet space coefficient could be used
            domain_to_domain_0[cell_minus] = domain_to_domain_0[cell_plus]
            # Same hack for the right submesh
            domain_to_domain_1[cell_plus] = domain_to_domain_1[cell_minus]

    return interface_entities, domain_to_domain_0, domain_to_domain_1

# Meshing utility functions
def mark_subdomains_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking subdomains of a unit square mesh with an interior square defined on [0.25, 0.75]^2.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y) = [0.25, 0.75]^2
        - tag value 2 : outer square, (x, y) = [0, 1]^2 \ [0.25, 0.75]^2
    
    """ 
    def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        """ Locator function for the inner square. """

        bool1 = np.logical_and(x[0] <= 0.75, x[0] >= 0.25) # True if inside inner box in x range
        bool2 = np.logical_and(x[1] <= 0.75, x[1] >= 0.25) # True if inside inner box in y range
        
        return np.logical_and(bool1, bool2)

    # Tag values
    INTRA = 1
    EXTRA = 2

    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells    = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker  = np.full(num_cells, EXTRA, dtype = np.int32)

    # Get all facets
    inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside)
    cell_marker[inner_cells] = INTRA

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking boundaries of a unit square mesh with an interior square defined on [0.25, 0.75]^2
    
    The boundaries have the following tags:
        - tag value 3 : outer boundary (\partial\Omega) 
        - tag value 4 : interface gamma between inner and outer square
        - tag value 5 : interior facets

    """    
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.75), y_range)

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.25), y_range)

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.25), x_range)

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.75), x_range)

    # Tag values
    PARTIAL_OMEGA = 3
    GAMMA         = 4
    DEFAULT       = 5

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = GAMMA

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = GAMMA

    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = GAMMA

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = GAMMA
    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)
    if len(facet_tags.find(GAMMA)) == 0:
        warnings.warn(f"No facets are marked with {GAMMA}")
    return facet_tags

def mark_boundaries_square_MMS(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit square mesh 
        with an interior square defined on [0.25, 0.75]^2.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.75), y_range)

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.25), y_range)

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.25), x_range)

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.75), x_range)

    # Tag values
    DEFAULT = 7
    PARTIAL_OMEGA   = 8
    LEFT    = 1
    RIGHT   = 2
    BOTTOM  = 3
    TOP     = 4

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = LEFT

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = RIGHT

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = TOP

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = BOTTOM

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def mark_subdomains_cube(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking subdomains of a unit cube mesh with an interior cube defined on [0.25, 0.75]^3.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y, z) = [0.25, 0.75]^3
        - tag value 2 : outer square, (x, y, z) = [0, 1]^3 \ [0.25, 0.75]^3
    
    """ 
    def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        """ Locator function for the inner square. """

        bool1 = np.logical_and(x[0] <= 0.75, x[0] >= 0.25) # True if inside inner box in x range
        bool2 = np.logical_and(x[1] <= 0.75, x[1] >= 0.25) # True if inside inner box in y range
        bool3 = np.logical_and(x[2] <= 0.75, x[2] >= 0.25) # True if inside inner box in z range
        
        inside_x_y =  np.logical_and(bool1, bool2)

        inside_cube = np.logical_and(bool3, inside_x_y)

        return inside_cube

    # Tag values
    INTRA = 1
    EXTRA = 2

    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells    = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker  = np.full(num_cells, EXTRA, dtype = np.int32)

    # Get all facets
    inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside)
    cell_marker[inner_cells] = INTRA

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_cube(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit cube mesh 
        with an interior cube defined on [0.25, 0.75]^3.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_right = np.isclose(x[0], 0.75)

        on_right = np.logical_and(x_right, np.logical_and(y_range, z_range))

        return on_right

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_left  = np.isclose(x[0], 0.25)

        on_left = np.logical_and(x_left, np.logical_and(y_range, z_range))

        return on_left

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_bottom = np.isclose(x[2], 0.25)
        
        on_bottom = np.logical_and(z_bottom, np.logical_and(x_range, y_range))

        return on_bottom

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_top    = np.isclose(x[2], 0.75)
        
        on_bottom = np.logical_and(z_top, np.logical_and(x_range, y_range))

        return on_bottom

    def front(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_front = np.isclose(x[1], 0.25)

        on_front = np.logical_and(y_front, np.logical_and(x_range, z_range))

        return on_front

    def back(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_back  = np.isclose(x[1], 0.75)

        on_back = np.logical_and(y_back, np.logical_and(x_range, z_range))

        return on_back

    # Tag values
    PARTIAL_OMEGA = 3
    GAMMA         = 4
    DEFAULT       = 5

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    # Get facets on the interface between the inner and outer cubes
    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = GAMMA

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = GAMMA

    front_facets = dfx.mesh.locate_entities(mesh, facet_dim, front)
    facet_marker[front_facets] = GAMMA

    back_facets = dfx.mesh.locate_entities(mesh, facet_dim, back)
    facet_marker[back_facets] = GAMMA

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = GAMMA

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = GAMMA

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def mark_boundaries_cube_MMS(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit cube mesh 
        with an interior cube defined on [0.25, 0.75]^3.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_right = np.isclose(x[0], 0.75)

        on_right = np.logical_and(x_right, np.logical_and(y_range, z_range))

        return on_right

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_left  = np.isclose(x[0], 0.25)

        on_left = np.logical_and(x_left, np.logical_and(y_range, z_range))

        return on_left

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_bottom = np.isclose(x[2], 0.25)
        
        on_bottom = np.logical_and(z_bottom, np.logical_and(x_range, y_range))

        return on_bottom

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_top    = np.isclose(x[2], 0.75)
        
        on_bottom = np.logical_and(z_top, np.logical_and(x_range, y_range))

        return on_bottom

    def front(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_front = np.isclose(x[1], 0.25)

        on_front = np.logical_and(y_front, np.logical_and(x_range, z_range))

        return on_front

    def back(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_back = np.isclose(x[1], 0.75)

        on_back = np.logical_and(y_back, np.logical_and(x_range, z_range))

        return on_back

    # Tag values
    DEFAULT = 7
    PARTIAL_OMEGA   = 8
    LEFT    = 1
    RIGHT   = 2
    FRONT   = 3
    BACK    = 4
    BOTTOM  = 5
    TOP     = 6

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    # Get facets on the interface between the inner and outer cubes
    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = LEFT

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = RIGHT

    front_facets = dfx.mesh.locate_entities(mesh, facet_dim, front)
    facet_marker[front_facets] = FRONT

    back_facets = dfx.mesh.locate_entities(mesh, facet_dim, back)
    facet_marker[back_facets] = BACK

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = TOP

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = BOTTOM

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags