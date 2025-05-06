import ufl

import numpy   as np
import dolfinx as dfx

from ufl       import grad, inner
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from CGx.utils.setup_mms         import SetupMMS, mark_MMS_boundaries
from CGx.utils.mixed_dim_problem import MixedDimensionalProblem

print = PETSc.Sys.Print

class ProblemEMI(MixedDimensionalProblem):

    def init(self):
        """ Constructor. """

        if self.MMS_test: self.setup_MMS_params() # Perform numerical verification
        
    def setup_spaces(self):

        print("Setting up function spaces ...")

        # Create a function space with continuous Lagrange elements
        P       = element("Lagrange", self.mesh.basix_cell(), self.fem_order)
        self.V  = dfx.fem.functionspace(self.mesh, P) # Space for functions defined on the entire mesh
        self.Vi = dfx.fem.functionspace(self.intra_mesh, P) # Intracellular space
        self.Ve = dfx.fem.functionspace(self.extra_mesh, P) # Extracellular space
        self.Vg = dfx.fem.functionspace(self.gamma_mesh, ("Lagrange", self.fem_order)) # Cellular membranes space (gamma)

        # Define block function space
        self.W = [self.Vi, self.Ve]

        # Create functions for storing the solutions
        self.wh = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Create functions for solution at previous timestep
        self.u_p = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Rename for more readable output
        self.u_p[0].name = "intra"
        self.u_p[1].name = "extra"

        # Store dof numbers
        self.num_dofs_Vi = self.Vi.dofmap.index_map.size_local

    def setup_boundary_conditions(self):

        if self.comm.rank == 0: print('Setting up boundary conditions ...')
        
        We = self.W[1] # Ease notation

        # Add Dirichlet boundary conditions on exterior boundary
        bce = []

        if self.dirichlet_bcs:

            facets_boundary = self.boundaries.indices[self.boundaries.values==self.boundary_tag]

            # Electric potential in extracellular space
            W_phi, _ = We.sub(self.N_ions).collapse()
            func = dfx.fem.Function(W_phi)

            if self.MMS_test:
                func.interpolate(self.phi_e_init)
            else:
                func.x.array[:] = self.phi_e_init

            dofs = dfx.fem.locate_dofs_topological((We.sub(self.N_ions), W_phi), self.boundaries.dim, facets_boundary)
            bc   = dfx.fem.dirichletbc(func, dofs, We.sub(self.N_ions))
            bce.append(bc)

        self.bcs = bce
            
    def setup_variational_form(self):
        
        print("Setting up variational form ...")

        # sanity check
        if len(self.ionic_models)==0 and self.comm.rank==0:
            raise RuntimeError('\nNo ionic model(s) specified.\nCall init_ionic_model() to provide ionic models.\n')

        # Aliases
        t   = self.t
        dt  = self.dt
        C_M = self.C_M
        phi_space = self.V

        # Source terms
        self.fi = dfx.fem.Function(self.W[0])
        self.fe = dfx.fem.Function(self.W[1])
        self.fg = dfx.fem.Function(self.Vg)

        # Set initial membrane potential
        if np.isclose(t.value, 0.0):
            self.phi_M_prev = dfx.fem.Function(phi_space)
            if self.MMS_test:
                self.phi_M_prev.interpolate(self.phi_M_init)
            else:
                self.phi_M_prev.x.array[:] = self.phi_M_init

        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags[0])
        
        # For the MMS test various gamma faces get different tags
        if self.MMS_test:
            # Create new boundary tags
            self.boundaries = mark_MMS_boundaries(self.mesh)
            dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.boundaries)
            if self.mesh.topology.dim == 2:
                # for Omega_i = [0.25, 0.75] x [0.25, 0.75]
                dS = dS((1, 2, 3, 4))
            elif self.mesh.topology.dim == 3:
                # for Omega_i = [0.25, 0.75] x [0.25, 0.75] x [0.25, 0.75]
                dS = dS((1, 2, 3, 4, 5, 6))

            dsOuter = dS(self.boundary_tag)

            # Update MMS expressions
            if self.dim == 2:
                src_terms, exact_sols, init_conds, bndry_terms = self.M.get_MMS_terms_KNPEMI_2D(t.value)
            elif self.dim == 3:
                src_terms, exact_sols, init_conds, bndry_terms = self.M.get_MMS_terms_KNPEMI_3D(t.value)

        # # init ionic models
        # for model in self.ionic_models:
        #     model._init()
        
        # Trial and test functions
        (ui, vi) = ufl.TrialFunction(self.W[0]), ufl.TestFunction(self.W[0])
        (ue, ve) = ufl.TrialFunction(self.W[1]), ufl.TestFunction(self.W[1])

        # Solutions at previous timestep
        ui_p = self.u_p[0]
        ue_p = self.u_p[1]     

        if np.isclose(t.value, 0.0):
            # First timestep
            # Set phi_e and phi_i just for visualization
            if self.MMS_test:
                ui_p.interpolate(self.phi_i_init)
                ue_p.interpolate(self.phi_e_init)
            else:
                ui_p.x.array[:] = self.phi_i_init
                ue_p.x.array[:] = self.phi_e_init
        
        # Initialize variational form block entries
        a00 = 0; a01 = 0; L0 = 0
        a10 = 0; a11 = 0; L1 = 0

        # Restrictions
        i_res = self.i_res
        e_res = self.e_res
        
        # Weak form - equation for phi_i
        a00 += dt * inner(self.sigma_i * grad(ui), grad(vi)) * dxi + C_M * inner(ui(i_res), vi(i_res)) * dS
        a01 -= C_M * inner(ue(e_res), vi(i_res)) * dS
        
        # Weak form - equation for phi_e
        a11 += dt * inner(self.sigma_e * grad(ue), grad(ve)) * dxe - C_M * inner(ue(e_res), ve(e_res)) * dS
        a10 -= C_M * inner(ui(i_res), ve(e_res)) * dS 

        # Linear form intracellular space
        L0 += dt * inner(self.fi, vi) * dxi + C_M * inner(self.fg, vi(i_res)) * dS
        
        # Linear form extracellular space
        L1 += dt * inner(self.fe, ve) * dxe + C_M * inner(self.fg, ve(e_res)) * dS

        # Store weak form in matrix and vector
        a = [[a00, a01],
             [a10, a11]]

        L = [L0, L1]

        # Convert to C++ forms
        self.a = dfx.fem.form(a, entity_maps=self.entity_maps, jit_options=self.jit_parameters)
        self.L = dfx.fem.form(L, entity_maps=self.entity_maps, jit_options=self.jit_parameters)

    def setup_preconditioner(self):

        if self.comm.rank == 0: print('Setting up preconditioner ...')

        # Aliases
        dt  = self.dt
        C_M = self.C_M

        # Integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags)
        
        # Trial and test functions
        (ui, vi) = ufl.TrialFunction(self.W[0]), ufl.TestFunction(self.W[0])
        (ue, ve) = ufl.TrialFunction(self.W[1]), ufl.TestFunction(self.W[1])

        # Initialize variational form
        p00 = 0
        p11 = 0

        # Setup diagonal preconditioner
        # Add flux contributions to weak form equations
        p00 += dt * inner(self.sigma_i * grad(ui), grad(vi)) * dxi - C_M * inner(ui(self.i_res), vi(self.i_res)) * dS
        p11 += dt * inner(self.sigma_e * grad(ue), grad(ve)) * dxe - C_M * inner(ue(self.e_res), ve(self.e_res)) * dS        

        # Create block preconditioner matrix
        P = [[p00, None],
             [None, p11]]

        # Convert to C++ form
        self.P = dfx.fem.form(P, entity_maps=self.entity_maps, jit_options=self.jit_parameters)

    def setup_MMS_params(self):

        self.MMS_test            = True
        self.dirichlet_bcs       = True
        self.mesh_conversion_factor = 1

        # ionic model
        self.HH_model = False

        self.C_M = 1
        self.F   = 1
        self.R   = 1
        self.T   = 1   
        self.psi = 1           

        self.M = SetupMMS(self.mesh)

        if self.mesh.geometry.dim == 2:
            src_terms, exact_sols, init_conds, bndry_terms = self.M.get_MMS_terms_KNPEMI_2D(0.0)
        elif self.mesh.geometry.dim == 3:
            src_terms, exact_sols, init_conds, bndry_terms = self.M.get_MMS_terms_KNPEMI_3D(0.0)
        
        # initial values
        self.phi_M_init = init_conds['phi_M']   # membrane potential (V)
        self.phi_i_init = exact_sols['phi_i_e'] # internal potential (V) just for visualization
        self.phi_e_init = exact_sols['phi_e_e'] # external potential (V) just for visualization

    def print_errors(self):
        
        # Get the exact solutions
        if self.dim == 2:
            _, exact_sols, _, _ = self.M.get_MMS_terms_KNPEMI_2D(self.t.value)
        elif self.dim == 3:
            _, exact_sols, _, _ = self.M.get_MMS_terms_KNPEMI_3D(self.t.value)

        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        phi_i = self.wh[0]
        phi_e = self.wh[1]
        
        err_phi_i = inner(phi_i - exact_sols['phi_i_e'], phi_i - exact_sols['phi_i_e']) * dxi
        err_phi_e = inner(phi_e - exact_sols['phi_e_e'], phi_e - exact_sols['phi_e_e']) * dxe

        # calculate local L2 error norms
        L2_err_phi_i = dfx.fem.assemble_scalar(dfx.fem.form(err_phi_i))
        L2_err_phi_e = dfx.fem.assemble_scalar(dfx.fem.form(err_phi_e))

        # Sum over processors and take square root to get global errors
        comm = self.comm # MPI communicator
        L2_err_phi_i = np.sqrt(comm.allreduce(L2_err_phi_i, op=MPI.SUM))
        L2_err_phi_e = np.sqrt(comm.allreduce(L2_err_phi_e,  op=MPI.SUM))

        # Print the errors
        print('#-------------- ERRORS --------------#')
        print(f"Hello from rank = {self.comm.rank}")
        print('L2 phi_i error:', L2_err_phi_i)
        print('L2 phi_e error:', L2_err_phi_e)

        self.errors = [L2_err_phi_i, L2_err_phi_e]        

    ### Default class variables ###
    # Mesh restrictions
    i_res = "+"
    e_res = "-"
    
    # Physical parameters
    # Mesh constants
    C_M = 0.1
    sigma_i = 1
    sigma_e = 1
    V_rest  = -0.065                 # resting membrane potential (V)

    # Initial conditions
    phi_e_init = 0         # external potential (V) (Constant)
    phi_i_init = -0.06774  # internal potential (V) just for visualization (Constant)
    phi_M_init = -0.06774  # membrane potential (V)	 (Constant)

    # Initial values of gating variables
    n_init_val = 0.27622914792
    m_init_val = 0.03791834627
    h_init_val = 0.68848921811

    # Mesh unit conversion factor
    mesh_conversion_factor = 1

    # Finite element polynomial order 
    fem_order = 1
    
    # Verification flag
    MMS_test        = False

    # Boundary condition type
    dirichlet_bcs   = False