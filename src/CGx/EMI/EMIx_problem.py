import ufl
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy   as np
import dolfinx as dfx

from ufl      import grad, inner
from mpi4py   import MPI
from petsc4py import PETSc
from CGx.EMI.EMIx_ionic_model    import g_syn_none, HH_model, Passive_model, IonicModel
from CGx.utils.setup_mms         import SetupMMS, mark_MMS_boundaries
from CGx.utils.mixed_dim_problem import MixedDimensionalProblem

print = PETSc.Sys.Print

class ProblemEMI(MixedDimensionalProblem):

    def init(self):
        """ Constructor. """

        if self.MMS_test: self.setup_MMS_params() # Perform numerical verification

    def add_ionic_model(self, model: IonicModel, tags: int | tuple | dict=None, stim_fun=g_syn_none):
        model = model[0]
        if model.__str__()=='Hodgkin-Huxley':
            model = HH_model(self, tags, stim_fun)
        elif model.__str__()=='Passive':
            model = Passive_model(self, tags)
        else:
            raise RuntimeError(f'Model type {model.__str__()} not supported. Choose either "HH" or "Passive".')

        self.ionic_models.append(model)
        
    def setup_spaces(self):

        print("Setting up function spaces ...")

        # Create a function space with continuous Lagrange elements
        self.V = dfx.fem.functionspace(self.mesh, ("Lagrange", self.fem_order))

        # Define block function space
        self.W = [self.V.clone(), self.V.clone()]

        # Create functions for storing the solutions
        self.wh = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Create functions for solution at previous timestep
        self.u_p = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Rename for more readable output
        self.wh[0].name  = "phi_i"
        self.wh[1].name  = "phi_e"
        self.u_p[0].name = "phi_i"
        self.u_p[1].name = "phi_e"

        print("Creating mesh restrictions ...")

        ### Restrictions
        # Get indices of the cells of the intra- and extracellular subdomains        
        if len(self.intra_tags) > 1:
            list_of_indices = [self.subdomains.find(tag) for tag in self.intra_tags]
            intra_indices = np.array([], dtype=np.int32)
            for l in list_of_indices:
                intra_indices = np.concatenate((intra_indices, l))
        else:
            intra_indices = self.subdomains.values==self.intra_tags[0]
        extra_indices = self.subdomains.values==self.extra_tag
        
        cells_intra = self.subdomains.indices[intra_indices]
        cells_extra = self.subdomains.indices[extra_indices]
        
        # Get interior and exterior dofs
        self.dofs_intra = dfx.fem.locate_dofs_topological(self.W[0], self.subdomains.dim, cells_intra)
        self.dofs_extra = dfx.fem.locate_dofs_topological(self.W[1], self.subdomains.dim, cells_extra)
        
        self.interior = multiphenicsx.fem.DofMapRestriction(self.W[0].dofmap, self.dofs_intra)
        self.exterior = multiphenicsx.fem.DofMapRestriction(self.W[1].dofmap, self.dofs_extra)

        self.restriction = [self.interior, self.exterior]

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
            
    def setup_bilinear_form(self):
        
        print("Setting up bilinear form ...")

        # sanity check
        if len(self.ionic_models)==0 and self.comm.rank==0:
            raise RuntimeError('\nNo ionic model(s) specified.\nCall init_ionic_model() to provide ionic models.\n')

        # Aliases
        t   = self.t
        dt  = self.dt
        C_M = self.C_M
        sigma_i = self.sigma_i
        sigma_e = self.sigma_e
        phi_space = self.V

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
        dS  = self.dS(self.gamma_tags)

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
        
        # Trial and test functions
        ui, vi = ufl.TrialFunction(self.W[0]), ufl.TestFunction(self.W[0])
        ue, ve = ufl.TrialFunction(self.W[1]), ufl.TestFunction(self.W[1])
        
        # Solutions at previous timestep
        ui_p = self.u_p[0]
        ue_p = self.u_p[1]     

        if np.isclose(t.value, 0.0):
            # First timestep
            # Set phi_e and phi_i just for visualization
            ui_p.x.array[:] = self.phi_M_init + self.phi_e_init
            ue_p.x.array[:] = self.phi_e_init
        
        # Weak form - equation for phi_i
        a00 = dt * inner(sigma_i * grad(ui), grad(vi)) * dxi + C_M * inner(ui('-'), vi('-')) * dS
        a01 = - C_M * inner(ue('+'), vi('-')) * dS
        
        # Weak form - equation for phi_e
        a11 = dt * inner(sigma_e * grad(ue), grad(ve)) * dxe + C_M * inner(ue('+'), ve('+')) * dS
        a10 = - C_M * inner(ui('-'), ve('+')) * dS 

        # Store weak form in matrix and vector
        a = [[a00, a01],
             [a10, a11]]

        # Compile bilinear form
        self.a = dfx.fem.form(a, jit_options=self.jit_parameters)

    def setup_linear_form(self):

        print('Setting up linear form ...')

        # Aliases
        dt = self.dt
        C_M = self.C_M
        source_i = self.source_i
        source_e = self.source_e
        t = float(self.t.value)
        
        # If at the first timestep, check that membrane potential
        # has been initialized with the bilinear form
        if np.isclose(t, 0.0): assert self.phi_M_prev is not None

        # Get integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags)

        # Test functions
        vi = ufl.TestFunction(self.W[0])
        ve = ufl.TestFunction(self.W[1])

        # Define source terms
        fi = inner(source_i, vi) * dxi
        fe = inner(source_e, ve) * dxe

        # Initialize dictionary for the ionic channels
        I_ch = dict.fromkeys(self.gamma_tags, 0)

        # Loop over ionic models and gamma tags and set the channel current
        for model in self.ionic_models:
            for gamma_tag in model.tags:
                I_ch[gamma_tag] = model._eval()
        
        # Loop over the gamma tags and add the source term contributions
        for gamma_tag in self.gamma_tags:
            fg = C_M*self.phi_M_prev - dt*I_ch[gamma_tag]

            fi += inner(fg, vi('-')) * dS(gamma_tag)
            fe -= inner(fg, ve('+')) * dS(gamma_tag)         

        L = [fi, fe]

        # Compile form
        self.L = dfx.fem.form(L, jit_options=self.jit_parameters)
        

    def setup_preconditioner(self):

        print('Setting up preconditioner ...')


        # Integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        
        # Trial and test functions
        ui, vi = ufl.TrialFunction(self.W[0]), ufl.TestFunction(self.W[0])
        ue, ve = ufl.TrialFunction(self.W[1]), ufl.TestFunction(self.W[1])

        # Setup diagonal preconditioner
        # Add flux contributions to weak form equations
        p00 = inner(self.sigma_i * grad(ui), grad(vi)) * dxi + inner(ui, vi) * dxi
        p11 = inner(self.sigma_e * grad(ue), grad(ve)) * dxe + inner(ue, ve) * dxe

        # Create block preconditioner matrix
        P = [[p00, None],
             [None, p11]]

        # Convert to C++ form
        self.P = dfx.fem.form(P, jit_options=self.jit_parameters)

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
    
    # Physical parameters
    # Mesh constants
    C_M = 0.1
    sigma_i = 1
    sigma_e = 1
    source_i = 0.0
    source_e = 0.0

    # Initial conditions
    phi_e_init = 0         # extracellular potential (V) (Constant)
    phi_M_init = -0.06774  # membrane      potential (V) (Constant)

    # Mesh unit conversion factor
    mesh_conversion_factor = 1

    # Finite element polynomial order 
    fem_order = 1
    
    # Verification flag
    MMS_test        = False

    # Boundary condition type
    dirichlet_bcs   = False