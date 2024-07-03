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

def calc_error_L2(u_h: dfx.fem.Function, u_exact: dfx.fem.Function, dX: ufl.Measure, degree_raise: int=3) -> float:
    """ Calculate the L2 error for a solution approximated with finite elements.

    Parameters
    ----------
    u_h : dolfinx.fem Function
        The solution function approximated with finite elements.

    u_exact : dolfinx.fem Function
        The exact solution function.

    degree_raise : int, optional
        The amount of polynomial degrees that the approximated solution
        is refined, by default 3

    Returns
    -------
    error_global : float
        The L2 error norm.
    """
    # Create higher-order function space for solution refinement
    degree = u_h.function_space.ufl_element().degree()
    family = u_h.function_space.ufl_element().family()
    mesh   = u_h.function_space.mesh

    if u_h.function_space.element.signature().startswith('Vector'):
        # Create higher-order function space based on vector elements
        W = dfx.fem.FunctionSpace(mesh, ufl.VectorElement(family=family, 
                                    degree=(degree+degree_raise), cell=mesh.ufl_cell()))
    else:
        # Create higher-order funciton space based on finite elements
        W = dfx.fem.FunctionSpace(mesh, (family, degree+degree_raise))

    # Interpolate the approximate solution into the refined space
    u_W = dfx.fem.Function(W)
    u_W.interpolate(u_h)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression
    u_exact_W = dfx.fem.Function(W)

    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
        u_exact_W.interpolate(u_expr)
    else:
        u_exact_W.interpolate(u_exact)
    
    # Compute the error in the higher-order function space
    e_W = dfx.fem.Function(W)
    with e_W.vector.localForm() as e_W_loc, u_W.vector.localForm() as u_W_loc, u_exact_W.vector.localForm() as u_ex_W_loc:
        e_W_loc[:] = u_W_loc[:] - u_ex_W_loc[:]

    # Integrate the error
    error        = dfx.fem.form(ufl.inner(e_W, e_W) * dX)
    error_local  = dfx.fem.assemble_scalar(error)

    return error_local


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