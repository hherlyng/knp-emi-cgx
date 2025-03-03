import ufl
import time
import numpy.typing

import numpy    as np
import scipy    as sp
import dolfinx  as dfx

from ufl       import inner, grad
from sys       import argv
from mpi4py    import MPI
from pathlib   import Path
from petsc4py  import PETSc
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_vector_block

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
    degree = u_h.function_space.ufl_element().degree
    family = u_h.function_space.ufl_element().element_family
    mesh   = u_h.function_space.mesh
    import basix.ufl
    if u_h.function_space.element.signature().startswith('Vector'):
        # Create higher-order function space based on vector elements
        W = dfx.fem.functionspace(mesh, basix.ufl.element(family=family, 
                                    degree=(degree+degree_raise), cell=mesh.basix_cell(), shape=(mesh.topology.dim, )))
    else:
        # Create higher-order funciton space based on finite elements
        W = dfx.fem.functionspace(mesh, (family, degree+degree_raise))

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
    with e_W.x.petsc_vec.localForm() as e_W_loc, u_W.x.petsc_vec.localForm() as u_W_loc, u_exact_W.x.petsc_vec.localForm() as u_ex_W_loc:
        e_W_loc[:] = u_W_loc[:] - u_ex_W_loc[:]

    # Integrate the error
    error        = dfx.fem.form(ufl.inner(e_W, e_W) * dX)
    error_local  = dfx.fem.assemble_scalar(error)

    return error_local
def transfer_meshtags_to_submesh(mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(len(sub_cell_to_parent), dtype=np.int32)
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = submesh.topology.index_map(
        entity_tag.dim).size_local + submesh.topology.index_map(entity_tag.dim).num_ghosts
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(child_vertices_as_parent, p_f_to_v.links(facet)).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
    tags =  dfx.mesh.meshtags(submesh, entity_tag.dim,
                                 np.arange(num_child_entities, dtype=np.int32), child_markers)
    tags.name = entity_tag.name
    return tags
def create_square_mesh_with_tags(N_cells: int,
                                 comm: MPI.Comm = MPI.COMM_WORLD,
                                 ghost_mode = dfx.mesh.GhostMode.shared_facet) \
        -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags, dfx.mesh.MeshTags)):
    """ Create a square mesh of a square within a square, with the inner square defined on (x, y) = [0.25, 0.75]^2.

    Parameters
    ----------
    N_cells : int
        number of mesh cells in x and y direction
    
    comm : MPI.Comm
        MPI communicator, by default MPI.COMM_WORLD
        
    ghost_mode : dfx.mesh.GhostMode
        mode that specifies how ghost nodes are shared in parallel, by default dfx.mesh.GhostMode.shared_facet

    Returns
    -------
    mesh : dfx.mesh.Mesh
        dolfinx mesh

    subdomains : dfx.mesh.MeshTags
        subdomain mesh tags

    boundaries : dfx.mesh.MeshTags
        boundary facet tags
    """

    mesh = dfx.mesh.create_unit_square(comm, N_cells, N_cells,
                                    cell_type = dfx.mesh.CellType.triangle,
                                    ghost_mode = ghost_mode)

    subdomains  = mark_subdomains_square(mesh)
    boundaries  = mark_boundaries_square(mesh)

    return mesh, subdomains, boundaries
def mark_subdomains_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking subdomains of a unit square mesh with an interior square defined on [0.25, 0.75]^2.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y) = [0.25, 0.75]^2
        - tag value 2 : outer square, (x, y) = [0, 1]^2 setminus [0.25, 0.75]^2
    
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
        - tag value 3 : outer boundary (partial Omega) 
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

    return facet_tags
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
def dump(thing, path):
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sp.sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data])

print = PETSc.Sys.Print
start_time = time.perf_counter()

# Subdomain tags
INTRA = 1
EXTRA = 2

# Boundary tags
PARTIAL_OMEGA = 3
GAMMA         = 4
DEFAULT       = 5

# Options for the fenicsx form compiler optimization
cache_dir       = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args"  : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

#----------------------------------------#
#     PARAMETERS AND SOLVER SETTINGS     #
#----------------------------------------#
# Space discretization parameters
N = int(argv[1])
P = 1

# Time discretization parameters
t          = 0.0
T          = 0.1
deltaT     = 0.01
time_steps = int(T / deltaT)

# Physical parameters
capacitance_membrane       = 1.0
conductivity_intracellular = 1.0
conductivity_extracellular = 1.0

# Flags
write_mesh     = False
save_output    = True    
save_matrix    = False
direct_solver  = True
ksp_type       = 'cg'
pc_type        = 'hypre'
ds_solver_type = 'mumps'
ksp_rtol       = 1e-8

# Timers
solve_time    = 0
assemble_time = 0

#------------------------------------#
#        FUNCTION EXPRESSIONS        #
#------------------------------------#
# Membrane potential expression
class InitialMembranePotential:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) #np.zeros_like(x[0])

# Forcing factor expressions
class IntracellularSource:
    def __init__(self, t_0: float):
        self.t = t_0 # Initial time

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return 8 * np.pi ** 2 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * (1 + np.exp(-self.t))

class ExtracellularSource:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return 8 * np.pi ** 2 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

# Exact solution expressions
class uiExact:
    def __init__(self, t_0: float):
        self.t = t_0

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]) * (1 + np.exp(-self.t))

class ueExact:
    def __init__(self): pass

    def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
        return np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

#-----------------------#
#          MESH         #
#-----------------------#
comm       = MPI.COMM_WORLD # MPI communicator
ghost_mode = dfx.mesh.GhostMode.shared_facet # How dofs are distributed in parallel

# Create mesh and submeshes
t1 = time.perf_counter()
mesh, ct, ft = create_square_mesh_with_tags(N_cells=N, comm=comm, ghost_mode=ghost_mode)
gdim = mesh.geometry.dim # Geometry dimension of the mesh
tdim = mesh.topology.dim # Topology dimension of the mesh, same as cell dimension
fdim = tdim-1 # Facet dimension
cells_intra  = ct.indices[ct.values==INTRA]
cells_extra  = ct.indices[ct.values==EXTRA]
gamma_facets = ft.indices[ft.values==GAMMA]
intra_sort = np.argsort(cells_intra)
extra_sort = np.argsort(cells_extra)
gamma_sort = np.argsort(gamma_facets)
intra_mesh, im_to_mesh, i_v_map, _ = dfx.mesh.create_submesh(msh=mesh, dim=gdim, entities=cells_intra[intra_sort])
extra_mesh, em_to_mesh, e_v_map, _ = dfx.mesh.create_submesh(msh=mesh, dim=gdim, entities=cells_extra[extra_sort])
gamma_mesh, gmesh_to_mesh, g_v_map, _ = dfx.mesh.create_submesh(msh=mesh, dim=fdim, entities=gamma_facets[gamma_sort])
print(f"Create mesh time: {time.perf_counter()-t1:.2f}")

# Mesh constants
dt      = dfx.fem.Constant(mesh, dfx.default_scalar_type(deltaT))
C_M     = dfx.fem.Constant(mesh, dfx.default_scalar_type(capacitance_membrane))
sigma_i = dfx.fem.Constant(mesh, dfx.default_scalar_type(conductivity_intracellular))
sigma_e = dfx.fem.Constant(mesh, dfx.default_scalar_type(conductivity_extracellular))

#------------------------------------------#
#     FUNCTION SPACES AND RESTRICTIONS     #
#------------------------------------------#
V  = dfx.fem.functionspace(mesh, ("Lagrange", P)) # Space for functions defined on the entire mesh
V1 = dfx.fem.functionspace(intra_mesh, ("Lagrange", P)) # Intracellular space
V2 = dfx.fem.functionspace(extra_mesh, ("Lagrange", P)) # Extracellular space
V3 = dfx.fem.functionspace(gamma_mesh, ("Lagrange", P)) # Cellular membrane (gamma)

num_dofs_V1 = V1.dofmap.index_map.size_local

print(f"MPI Rank {comm.rank}")
print(f"Size of local index map V-intra: {V1.dofmap.index_map.size_local}")
print(f"Size of global index map V-intra: {V1.dofmap.index_map.size_global}")
print(f"Number of ghost nodes V-intra: {V1.dofmap.index_map.num_ghosts}\n")
print(f"Size of local index map V-extra: {V2.dofmap.index_map.size_local}")
print(f"Size of global index map V-extra: {V2.dofmap.index_map.size_global}")
print(f"Number of ghost nodes V-extra: {V2.dofmap.index_map.num_ghosts}\n")
print(f"Size of local index map V-gamma: {V3.dofmap.index_map.size_local}")
print(f"Size of global index map V-gamma: {V3.dofmap.index_map.size_global}")
print(f"Number of ghost nodes V-gamma: {V3.dofmap.index_map.num_ghosts}")

# Trial and test functions
ui, ue = ufl.TrialFunction(V1), ufl.TrialFunction(V2)
vi, ve = ufl.TestFunction (V1), ufl.TestFunction (V2)

# Functions for storing the solutions and the exact solutions
ui_h, ui_ex = dfx.fem.Function(V1), dfx.fem.Function(V1)
ue_h, ue_ex = dfx.fem.Function(V2), dfx.fem.Function(V2)

ui_ex_expr = uiExact(t_0=t)
ui_ex.interpolate(ui_ex_expr)

ue_ex_expr = ueExact()
ue_ex.interpolate(ue_ex_expr)

# Membrane potential function
v_expr = InitialMembranePotential()
v = dfx.fem.Function(V3)
v.interpolate(v_expr)

# Membrane forcing term function
fg = dfx.fem.Function(V3)

# Forcing term in the intracellular space
fi_expr = IntracellularSource(t_0=t)
fi = dfx.fem.Function(V1)
fi.interpolate(fi_expr)

# Forcing term in the extracellular space
fe_expr = ExtracellularSource()
fe = dfx.fem.Function(V2)
fe.interpolate(fe_expr)

print(f"Time 1: {time.perf_counter() - start_time:.2f}")

t_second = time.perf_counter()
#------------------------------------#
#        VARIATIONAL PROBLEM         #
#------------------------------------#


# mesh will be used as the integration domain, so need to create
# mapping from cells in mesh to the cells in intra_mesh and extra_mesh
mesh_cell_imap = mesh.topology.index_map(tdim)
num_cells = mesh_cell_imap.size_local + mesh_cell_imap.num_ghosts
mesh_to_im = np.full(num_cells, -1, dtype=np.int32)
mesh_to_em = np.full(num_cells, -1, dtype=np.int32)
mesh_to_im[im_to_mesh] = np.arange(len(im_to_mesh), dtype=np.int32)
mesh_to_em[em_to_mesh] = np.arange(len(em_to_mesh), dtype=np.int32)
mesh_facet_imap = mesh.topology.index_map(tdim-1)
num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts

mesh_to_gm = np.full(num_facets, -1, dtype=np.int32)
mesh_to_gm[gmesh_to_mesh] = np.arange(len(gmesh_to_mesh), dtype=np.int32)
entity_maps = {intra_mesh : mesh_to_im,
               extra_mesh : mesh_to_em,
               gamma_mesh : mesh_to_gm}

# Compute integration entities for gamma (cellular membrane)
gamma_entities, mesh_to_im, mesh_to_em = compute_interface_integration_entities(msh=mesh, interface_facets=gamma_facets,
                            domain_0_cells=cells_intra, domain_1_cells=cells_extra, domain_to_domain_0=mesh_to_im, domain_to_domain_1=mesh_to_em)
# Define integral measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct) # Cell integrals
dS = ufl.Measure("dS", domain=mesh, subdomain_data=[(GAMMA, gamma_entities)]) # Facet integrals
dS = dS(GAMMA) # Restrict facet integrals to gamma interface

# Define restrictions
i_res = '+'
e_res = '-'
vg = ui_h(i_res) - ue_h(e_res) # Membrane potential

# First row of block bilinear form
a11 = dt * inner(sigma_i * grad(ui), grad(vi)) * dx(INTRA) + C_M * inner(ui(i_res), vi(i_res)) * dS # ui terms
a12 = - C_M * inner(ue(e_res), vi(i_res)) * dS # ue terms

# Second row of block bilinear form
a21 = - C_M * inner(ui(i_res), ve(e_res)) * dS # ui terms
a22 = dt * inner(sigma_e * grad(ue), grad(ve)) * dx(EXTRA) + C_M * inner(ue(e_res), ve(e_res)) * dS # ue terms

# Define boundary conditions
zero = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0)) # Grounded exterior boundary BC

facets_partial_Omega = ft.indices[ft.values==PARTIAL_OMEGA] # Get indices of the facets on the exterior boundary
extra_mesh.topology.create_connectivity(fdim, tdim)
ft2 = transfer_meshtags_to_submesh(mesh, ft, extra_mesh, e_v_map, em_to_mesh)

dofs_V2_partial_Omega = dfx.fem.locate_dofs_topological(V2, fdim, ft2.find(PARTIAL_OMEGA)) # Get the dofs on the exterior boundary facets
bce = dfx.fem.dirichletbc(zero, dofs_V2_partial_Omega, V2) # Set Dirichlet BC on exterior boundary facets

bcs = [bce]

# Compile block form
a11 = dfx.fem.form(a11, entity_maps=entity_maps, jit_options=jit_parameters)
a12 = dfx.fem.form(a12, entity_maps=entity_maps, jit_options=jit_parameters)
a21 = dfx.fem.form(a21, entity_maps=entity_maps, jit_options=jit_parameters)
a22 = dfx.fem.form(a22, entity_maps=entity_maps, jit_options=jit_parameters)
a_cpp = [[a11, a12],
         [a21, a22]]


#---------------------------#
#      MATRIX ASSEMBLY      #
#---------------------------#
t1 = time.perf_counter() # Timestamp for assembly time-lapse

# Assemble the block linear system matrix
A = assemble_matrix_block(a_cpp, bcs=bcs)
A.assemble()
assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time

# Create RHS and solution vectors
L_cpp = [
    dfx.fem.form(dt * inner(fi, vi) * dx(INTRA) + C_M * inner(fg, vi('-')) * dS, 
        entity_maps=entity_maps, jit_options=jit_parameters), 
    dfx.fem.form(dt * inner(fe, ve) * dx(EXTRA) - C_M * inner(fg, ve('+')) * dS, 
        entity_maps=entity_maps, jit_options=jit_parameters)
]
b = assemble_vector_block(L_cpp, a=a_cpp, bcs=bcs) # Assemble RHS vector
sol_vec = create_vector_block(L_cpp) # Create solution vector


# Configure Krylov solver
ksp = PETSc.KSP()
ksp.create(comm)
ksp.setOperators(A)
ksp.setTolerances(rtol=ksp_rtol)

# Set options based on direct/iterative solution method
if direct_solver:
    print("Using a direct solver ...")
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(ds_solver_type)
else:
    print("Using an iterative solver ...")
    opt = PETSc.Options()
    opt['ksp_converged_reason'] = None
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)

ksp.setFromOptions()

if save_output:
    # Create output files
    out_ui = dfx.io.XDMFFile(mesh.comm, "ui_submesh_square.xdmf", "w")
    out_ue = dfx.io.XDMFFile(mesh.comm, "ue_submesh_square.xdmf", "w")
    out_v  = dfx.io.XDMFFile(mesh.comm, "v_submesh_square.xdmf" , "w")

    out_ui.write_mesh(intra_mesh)
    out_ue.write_mesh(extra_mesh)
    out_v.write_mesh(gamma_mesh)
    out_v.write_function(v)


print(f"Time 2: {time.perf_counter()-t_second:.2f}")

#---------------------------------#
#        SOLUTION TIMELOOP        #
#---------------------------------#
for i in range(time_steps):

    # Increment time
    t += deltaT

    # Update time-dependent expressions
    fi_expr.t = t
    fi.interpolate(fi_expr)

    # Update and assemble vector that is the RHS of the linear system
    t1 = time.perf_counter() # Timestamp for assembly time-lapse

    # Forcing term on membrane
    if i==0:
        fg = v - dt / capacitance_membrane * v
    else:
        fg = vg - dt / capacitance_membrane * vg
    Li = dt * inner(fi, vi) * dx(INTRA) + C_M * inner(fg, vi(i_res)) * dS # Linear form intracellular space
    Le = dt * inner(fe, ve) * dx(EXTRA) - C_M * inner(fg, ve(e_res)) * dS # Linear form extracellular space

    # Compile block linear form
    L_cpp = [dfx.fem.form(Li, entity_maps=entity_maps, jit_options=jit_parameters),
             dfx.fem.form(Le, entity_maps=entity_maps, jit_options=jit_parameters)] 
        
    with b.localForm() as b_loc: b_loc.set(0.0) # Avoid accumulation of values
    assemble_vector_block(b, L=L_cpp, a=a_cpp, bcs=bcs) # Assemble RHS vector
    
    assemble_time += time.perf_counter() - t1 # Add time lapsed to total assembly time
    
    # Solve the system
    t1 = time.perf_counter() # Timestamp for solver time-lapse
    ksp.solve(b, sol_vec)

    # Update ghost values
    sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ui_h.x.array[:] = sol_vec.array[:num_dofs_V1]
    ue_h.x.array[:] = sol_vec.array[num_dofs_V1:]
    vg = ui_h(i_res) - ue_h(e_res)

    solve_time += time.perf_counter() - t1 # Add time lapsed to total solver time
    
    if save_output:
        out_ui.write_function(ui_h, t)
        out_ue.write_function(ue_h, t)

# Update time of exact ui expression
ui_ex_expr.t = t
ui_ex.interpolate(ui_ex_expr)

#------------------------------#
#         POST PROCESS         #
#------------------------------#
# Error analysis
L2_error_i_local = calc_error_L2(u_h=ui_h, u_exact=ui_ex, dX=ufl.Measure("dx", domain=intra_mesh)) # Local L2 error (squared) of intracellular electric potential
L2_error_e_local = calc_error_L2(u_h=ue_h, u_exact=ue_ex, dX=ufl.Measure("dx", domain=extra_mesh)) # Local L2 error (squared) of extracellular electric potential

L2_error_i_global = np.sqrt(comm.allreduce(L2_error_i_local, op=MPI.SUM)) # Global L2 error of intracellular electric potential
L2_error_e_global = np.sqrt(comm.allreduce(L2_error_e_local, op=MPI.SUM)) # Global L2 error of extracellular electric potential

# Sum local assembly and solve times to get global values
max_local_assemble_time = comm.allreduce(assemble_time, op=MPI.MAX) # Global assembly time
max_local_solve_time    = comm.allreduce(solve_time   , op=MPI.MAX) # Global solve time

# Print stuff
print("\n#-----------INFO-----------#\n")
print("MPI size = ", comm.size)
print("dt = ", deltaT)
print("N = ", N)
print("P = ", P)

print("\n#----------ERRORS----------#\n")
print(f"L2 error norm intracellular: {L2_error_i_global:.2e}")
print(f"L2 error norm extracellular: {L2_error_e_global:.2e}")

print("\n#-------TIME ELAPSED-------#\n")
print(f"Max assembly time: {max_local_assemble_time:.3f} seconds\n")
print(f"Max solve time: {max_local_solve_time:.3f} seconds\n")

print("#--------------------------#")

print(f"Script time elapsed: {time.perf_counter() - start_time:.3f} seconds")

# Write solutions to file
if save_output:
    out_ui.close()
    out_ue.close()

if write_mesh:
    with dfx.io.XDMFFile(mesh.comm, "dfx_square_mesh.xdmf", "w") as mesh_file:
        mesh_file.write_mesh(mesh)
        ft.name = "ft"
        ct.name = "ct"
        mesh_file.write_meshtags(ft, mesh.geometry)
        mesh_file.write_meshtags(ct, mesh.geometry)