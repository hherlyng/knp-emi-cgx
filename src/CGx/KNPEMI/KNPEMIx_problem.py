import ufl
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy   as np
import dolfinx as dfx

from ufl      import grad, inner, dot
from mpi4py   import MPI
from petsc4py import PETSc
from CGx.utils.setup_mms         import SetupMMS, mark_MMS_boundaries
from CGx.utils.mixed_dim_problem import MixedDimensionalProblem

print = PETSc.Sys.Print

class ProblemKNPEMI(MixedDimensionalProblem):

    def init(self):
        """ Constructor. """

        if self.MMS_test: self.setup_MMS_params() # Perform numerical verification
        
    def setup_spaces(self):

        print("Setting up function spaces ...")

        # Define elements
        P = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), self.fem_order)

        # Ion concentrations for each ion + electric potential
        element_list = [P] * (self.N_ions + 1)

        self.V = dfx.fem.FunctionSpace(self.mesh, ufl.MixedElement(element_list))

        # Define block function space
        V1 = self.V.clone()
        V2 = self.V.clone()
        self.W = [V1, V2]

        # Create functions for storing the solutions
        self.wh = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Create functions for solution at previous timestep
        self.u_p = [dfx.fem.Function(self.W[0]), dfx.fem.Function(self.W[1])]

        # Rename for more readable output
        self.u_p[0].name = "intra"
        self.u_p[1].name = "extra"

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

        print('Setting up boundary conditions ...')
        
        We = self.W[1] # Ease notation

        # Add Dirichlet boundary conditions on exterior boundary
        bce = []

        if self.dirichlet_bcs:

            facets_boundary = self.boundaries.indices[self.boundaries.values==self.boundary_tag]

            # BCs for concentrations
            for idx, ion in enumerate(self.ion_list):

                We_ion, _ = We.sub(idx).collapse()
                func = dfx.fem.Function(We_ion)
                if self.MMS_test:
                    func.interpolate(ion['ke_init'])
                else:
                    func.x.array[:] = ion['ke_init']

                dofs = dfx.fem.locate_dofs_topological((We.sub(idx), We_ion), self.boundaries.dim, facets_boundary)
                bc   = dfx.fem.dirichletbc(func, dofs, We.sub(idx))
                bce.append(bc)

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


    def setup_constants(self, dt: float):
        """ Initialize constants as dolfinx.fem.Constant objects to reduce total compilation time. """

        # TODO
        self.dt       = dfx.fem.Constant(self.mesh, PETSc.ScalarType(dt))
        self.F_const  = dfx.fem.Constant(self.mesh, PETSc.ScalarType(self.F))
        self.psi_const = dfx.fem.Constant(self.mesh, PETSc.ScalarType(self.psi))
        self.C_M_const = dfx.fem.Constant(self.mesh, PETSc.ScalarType(self.C_M))
        self.domain_L_const = dfx.fem.Constant(self.mesh, PETSc.ScalarType(self.domain_L))
        for ion in self.ion_list:
            # Get ion attributes
            ion['z']  = dfx.fem.Constant(self.mesh, PETSc.ScalarType(ion['z']))
            ion['Di'] = dfx.fem.Constant(self.mesh, PETSc.ScalarType(ion['Di']))
            ion['De'] = dfx.fem.Constant(self.mesh, PETSc.ScalarType(ion['De']))
            
    def setup_variational_form(self):

        # sanity check
        if len(self.ionic_models)==0 and self.comm.rank==0:
            raise RuntimeError('\nNo ionic model(s) specified.\nCall init_ionic_model() to provide ionic models.\n')

        # Aliases
        dt  = self.dt
        F   = self.F
        psi = self.psi
        C_M = self.C_M
        t   = self.t
        phi_space, _ = self.V.sub(self.N_ions).collapse()

        # Set initial membrane potential
        if np.isclose(t.value, 0.0):
            self.phi_M_prev = dfx.fem.Function(phi_space)
            if self.MMS_test:
                self.phi_M_prev.interpolate(self.phi_M_init)
            else:
                self.phi_M_prev.x.array[:] = self.phi_M_init
        
        print("Setting up variational form ...")

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
        
            # create ions
            self.Na = {'Di':1.0, 'De':1.0, 'z':1.0,
                    'ki_init':init_conds['Na_i'],
                    'ke_init':init_conds['Na_e'],		
                    'k_scaling': 1,	  
                    'f_k_i':src_terms['f_Na_i'],
                    'f_k_e':src_terms['f_Na_e'],
                    'J_k_e':bndry_terms['J_Na_e'],
                    'phi_i_e':exact_sols['phi_i_e'],
                    'phi_e_e':exact_sols['phi_e_e'],
                    'f_phi_i':src_terms['f_phi_i'],
                    'f_phi_e':src_terms['f_phi_e'],
                    'f_g_M':src_terms['f_g_M'],
                    'f_I_M':src_terms['f_I_M'],
                    'name':'Na'}

            self.K = {'Di':1.0, 'De':1.0, 'z':1.0,
                    'ki_init':init_conds['K_i'],
                    'ke_init':init_conds['K_e'],			 
                    'k_scaling': 1,
                    'f_k_i':src_terms['f_K_i'],
                    'f_k_e':src_terms['f_K_e'],
                    'J_k_e':bndry_terms['J_K_e'],
                    'phi_i_e':exact_sols['phi_i_e'],
                    'phi_e_e':exact_sols['phi_e_e'],
                    'f_phi_i':src_terms['f_phi_i'],
                    'f_phi_e':src_terms['f_phi_e'],
                    'f_g_M':src_terms['f_g_M'],
                    'f_I_M':src_terms['f_I_M'],
                    'name':'K'}

            self.Cl = {'Di':1.0, 'De':1.0, 'z':-1.0,
                    'ki_init':init_conds['Cl_i'],
                    'ke_init':init_conds['Cl_e'],		
                    'k_scaling': 1,	  
                    'f_k_i':src_terms['f_Cl_i'],
                    'f_k_e':src_terms['f_Cl_e'],
                    'J_k_e':bndry_terms['J_Cl_e'],
                    'phi_i_e':exact_sols['phi_i_e'],
                    'phi_e_e':exact_sols['phi_e_e'],
                    'f_phi_i':src_terms['f_phi_i'],
                    'f_phi_e':src_terms['f_phi_e'],
                    'f_g_M':src_terms['f_g_M'],
                    'f_I_M':src_terms['f_I_M'],
                    'name':'Cl'}

            # create ion list
            self.ion_list = [self.Na, self.K, self.Cl]

        # init ionic models
        for model in self.ionic_models:
            model._init()
        
        # Trial and test functions
        (ui, vi) = ufl.TrialFunctions(self.W[0]), ufl.TestFunctions(self.W[0])
        (ue, ve) = ufl.TrialFunctions(self.W[1]), ufl.TestFunctions(self.W[1])

        # Solutions at previous timestep
        ui_p = self.u_p[0]
        ue_p = self.u_p[1]

        # Intracellular potential
        phi_i  = ui[self.N_ions] # trial function
        vphi_i = vi[self.N_ions] # test function

        # Extracellular potential
        phi_e  = ue[self.N_ions] # trial function
        vphi_e = ve[self.N_ions] # test function         

        # Initialize
        alpha_i_sum = 0 # Sum of fractions intracellular
        alpha_e_sum = 0 # Sum of fractions extracellular
        J_phi_i     = 0 # Total intracellular flux
        J_phi_e     = 0 # Total extracellular flux

        # total channel current
        I_ch = dict.fromkeys(self.gamma_tags, 0)

        # Initialize parts of variational formulation
        for idx, ion in enumerate(self.ion_list):

            # Get ion attributes
            z  = ion['z']
            Di = ion['Di']
            De = ion['De']

            # Set initial value of intra- and extracellular ion concentrations
            if np.isclose(t.value, 0.0):
                if self.MMS_test:
                    ui_p.sub(idx).interpolate(ion['ki_init'])
                    ue_p.sub(idx).interpolate(ion['ke_init'])
                else:
                    # Get dof mapping between subspace and parent space
                    _, sub_to_parent_i = ui_p.sub(idx).function_space.collapse()
                    _, sub_to_parent_e = ue_p.sub(idx).function_space.collapse()

                    # Set the array values at the subspace dofs 
                    ui_p.sub(idx).x.array[sub_to_parent_i] = ion['ki_init']
                    ue_p.sub(idx).x.array[sub_to_parent_e] = ion['ke_init']

            # Add ion specific contribution to fraction alpha
            alpha_i_sum += Di * z**2 * ui_p.sub(idx)
            alpha_e_sum += De * z**2 * ue_p.sub(idx)
            
            # Calculate and update Nernst potential for current ion
            ion['E'] = (psi/z) * ufl.ln(ue_p.sub(idx) / ui_p.sub(idx))

            # Initialize dictionary of ionic channel
            ion['I_ch'] = dict.fromkeys(self.gamma_tags)

            # Loop over ionic models
            for model in self.ionic_models:

                # Loop over ionic model tags
                for gamma_tag in model.tags:

                    ion['I_ch'][gamma_tag] = model._eval(idx)

                    # Add contribution to total channel current
                    I_ch[gamma_tag] += ion['I_ch'][gamma_tag]
            
        if np.isclose(t.value, 0.0):
            # First timestep
            # Set phi_e and phi_i just for visualization
            if self.MMS_test:
                ui_p.sub(self.N_ions).interpolate(self.phi_i_init)
                ue_p.sub(self.N_ions).interpolate(self.phi_e_init)
            else:
                # Get dof mapping between subspace and parent space
                _, sub_to_parent_i = ui_p.sub(self.N_ions).function_space.collapse()
                _, sub_to_parent_e = ue_p.sub(self.N_ions).function_space.collapse()
                
                # Set the array values at the subspace dofs 
                ui_p.sub(self.N_ions).x.array[sub_to_parent_i] = self.phi_i_init
                ue_p.sub(self.N_ions).x.array[sub_to_parent_e] = self.phi_e_init
        
        # Initialize variational form block entries
        a00 = 0; a01 = 0; L0 = 0
        a10 = 0; a11 = 0; L1 = 0
       
        # Setup ion-specific part of variational formulation
        for idx, ion in enumerate(self.ion_list):
            
            # Get ion attributes
            z  = ion['z']
            Di = ion['Di']
            De = ion['De']
            I_ch_k = ion['I_ch']

            # Set intracellular ion attributes
            ki  = ui[idx]       # Trial function
            vki = vi[idx]       # Test function
            ki_prev = ui_p.sub(idx) # Previous solution

            # Set extracellular ion attributes
            ke  = ue[idx]       # Trial function
            vke = ve[idx]       # Test function
            ke_prev = ue_p.sub(idx) # Previous solution

            # Set fraction of ion-specific intra- and extracellular I_cap
            alpha_i = Di * z**2 * ki_prev / alpha_i_sum
            alpha_e = De * z**2 * ke_prev / alpha_e_sum

            # Linearized ion fluxes
            Ji = - Di * grad(ki) - (Di*z/psi) * ki_prev * grad(phi_i)
            Je = - De * grad(ke) - (De*z/psi) * ke_prev * grad(phi_e)

            # Some useful constants
            C_i = C_M * alpha_i('-') / (F*z)
            C_e = C_M * alpha_e('-') / (F*z)

            # Weak form - equation for k_i
            a00 += ki*vki*dxi - dt * inner(Ji, grad(vki)) * dxi
            a00 +=   C_i * inner(phi_i('-'), vki('-')) * dS
            a01 += - C_i * inner(phi_e('-'), vki('-')) * dS
            L0  += ki_prev*vki*dxi
            
            # Weak form - equation for k_e
            a11 += ke*vke*dxe - dt * inner(Je, grad(vke)) * dxe
            a11 +=   C_e * inner(phi_e('-'), vke('-')) * dS 
            a10 += - C_e * inner(phi_i('-'), vke('-')) * dS 
            L1  += ke_prev*vke*dxe

            # Ionic channels
            for gamma_tag in self.gamma_tags:
                L0 -= (dt*I_ch_k[gamma_tag] - alpha_i('-')*C_M*self.phi_M_prev) / (F*z) * vki('-') * dS(gamma_tag)
                L1 += (dt*I_ch_k[gamma_tag] - alpha_e('-')*C_M*self.phi_M_prev) / (F*z) * vke('-') * dS(gamma_tag)

            # Add contributions to total current flux
            J_phi_i += z*Ji
            J_phi_e += z*Je

            # Source terms
            L0 += inner(ion['f_i'], vki) * dxi
            L1 += inner(ion['f_e'], vke) * dxe
            
            if self.MMS_test:
                # Define outward normal on exterior boundary (\partial\Omega)
                n_outer = ufl.FacetNormal(self.mesh)

                # Concentrations source terms
                L0 += dt * inner(ion['f_k_i'], vki) * dxi # Equation for k_i
                L1 += dt * inner(ion['f_k_e'], vke) * dxe # Equation for k_e

                # Enforcing correction for I_m
                for i, JM in enumerate(ion['f_I_M'], 1):
                    L0 += dt/(F*z) * alpha_i('-') * inner(JM, vki('-')) * dS(i)
                    L1 -= dt/(F*z) * alpha_e('-') * inner(JM, vke('-')) * dS(i)
                
                # Enforcing correction for I_m, assuming gM_k = gM / N_ions
                L1 -= dt/(F*z) * sum(alpha_e('-')*inner(gM, vke('-'))*dS(i) for i, gM in enumerate(ion['f_g_M'], 1))

                # Exterior boundary terms (zero in "physical problem")
                L1 -=  dt * inner(dot(ion['J_k_e'], n_outer('-')), vke('-')   ) * dsOuter # Equation for k_e
                L1 += F*z * inner(dot(ion['J_k_e'], n_outer('-')), vphi_e('-')) * dsOuter # Equation for phi_e
        
        # Weak form - equation for phi_i
        a00 -= inner(J_phi_i, grad(vphi_i)) * dxi - (C_M/(F*dt)) * inner(phi_i('-'), vphi_i('-')) * dS
        a01 -= (C_M/(F*dt)) * inner(phi_e('-'), vphi_i('-')) * dS
        
        # Weak form - equation for phi_e
        a11 -= inner(J_phi_e, grad(vphi_e)) * dxe - (C_M/(F*dt)) * inner(phi_e('-'), vphi_e('-')) * dS
        a10 -= (C_M/(F*dt)) * inner(phi_i('-'), vphi_e('-')) * dS 
        
        for gamma_tag in self.gamma_tags:
            L0  -= (1/F) * (I_ch[gamma_tag] - C_M*self.phi_M_prev/dt) * vphi_i('-') * dS(gamma_tag)
            L1  += (1/F) * (I_ch[gamma_tag] - C_M*self.phi_M_prev/dt) * vphi_e('-') * dS(gamma_tag)
        
        if self.MMS_test:
            # Phi source terms
            L0 -= inner(ion['f_phi_i'], vphi_i) * dxi # Equation for phi_i
            L1 -= inner(ion['f_phi_e'], vphi_e) * dxe # Equation for phi_e

            # Enforcing correction for I_m
            for i, JM in enumerate(ion['f_I_M'], 1):
                L0 += inner(JM, vphi_i('-')) * dS(i)
                L1 -= inner(JM, vphi_e('-')) * dS(i)

            L1 -= sum(inner(gM, vphi_e('-')) * dS(i) for i, gM in enumerate(ion['f_g_M'], 1))
              

        # Store weak form in matrix and vector
        a = [[a00, a01],
             [a10, a11]]

        L = [L0, L1]

        # Convert to C++ forms
        self.a = dfx.fem.form(a, jit_options=self.jit_parameters)
        self.L = dfx.fem.form(L, jit_options=self.jit_parameters)
        

    def setup_preconditioner(self, use_block_jacobi: bool):

        print('Setting up preconditioner ...')

        # Aliases
        dt  = self.dt
        F   = self.F
        psi = self.psi
        C_M = self.C_M

        ui_p = self.u_p[0]
        ue_p = self.u_p[1]

        # Integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags)
        
        # Trial and test functions
        (ui, vi) = ufl.TrialFunctions(self.W[0]), ufl.TestFunctions(self.W[0])
        (ue, ve) = ufl.TrialFunctions(self.W[1]), ufl.TestFunctions(self.W[1])

        # Intracellular electric potential
        phi_i  = ui[self.N_ions]
        vphi_i = vi[self.N_ions]

        # Extracellular electric potential
        phi_e  = ue[self.N_ions]
        vphi_e = ve[self.N_ions]
        
        # Initialize fluxes
        J_phi_i = 0
        J_phi_e = 0

        # Initialize variational form
        p00 = 0
        p11 = 0

        # Setup diagonal preconditioner
        for idx, ion in enumerate(self.ion_list):

            # Get ion attributes
            z    = ion['z']
            Di   = ion['Di']
            De   = ion['De']

            # Set intracellular ion attributes
            ki  = ui[idx]       # Trial function
            vki = vi[idx]       # Test function
            ki_prev = ui_p.sub(idx) # Previous solution

            # Set extracellular ion attributes
            ke  = ue[idx]       # Trial function
            vke = ve[idx]       # Test function
            ke_prev = ue_p.sub(idx) # Previous solution

            p00 += dt * inner(Di*grad(ki), grad(vki)) * dxi + ki*vki*dxi
            p11 += dt * inner(De*grad(ke), grad(vke)) * dxe + ke*vke*dxe

            # Add contribution to total current flux
            if use_block_jacobi:
                Ji = - Di*z/psi * ki_prev*grad(phi_i)
                Je = - De*z/psi * ke_prev*grad(phi_e)
            else:
                Ji = - Di*grad(ki) - Di*z/psi * ki_prev*grad(phi_i)
                Je = - De*grad(ke) - De*z/psi * ki_prev*grad(phi_e)
            
            # Add contribution to total flux
            J_phi_i += z*Ji
            J_phi_e += z*Je

            # weak form - equation for k_i
            p00 += ki*vki*dxi + dt * inner(Di*grad(ki), grad(vki)) * dxi

            # weak form - equation for k_e
            p11 += ke*vke*dxe + dt * inner(De*grad(ke), grad(vke)) * dxe

        # Add flux contributions to weak form equations
        p00 -= inner(J_phi_i, grad(vphi_i)) * dxi - (C_M/(F*dt)) * inner(phi_i('-'), vphi_i('-')) * dS
        p11 -= inner(J_phi_e, grad(vphi_e)) * dxe - (C_M/(F*dt)) * inner(phi_e('-'), vphi_e('-')) * dS        

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
        self.phi_i_init = exact_sols['phi_i_e'] # internal potential (V) just for visualization
        self.phi_e_init = exact_sols['phi_e_e'] # external potential (V) just for visualization

        # create ions
        self.Na = {'Di':1.0, 'De':1.0, 'z':1.0,
                'ki_init':init_conds['Na_i'],
                'ke_init':init_conds['Na_e'],		
                'k_scaling': 1,	  
                'f_k_i':src_terms['f_Na_i'],
                'f_k_e':src_terms['f_Na_e'],
                'J_k_e':bndry_terms['J_Na_e'],
                'phi_i_e':exact_sols['phi_i_e'],
                'phi_e_e':exact_sols['phi_e_e'],
                'f_phi_i':src_terms['f_phi_i'],
                'f_phi_e':src_terms['f_phi_e'],
                'f_g_M':src_terms['f_g_M'],
                'f_I_M':src_terms['f_I_M'],
                'name':'Na'}

        self.K = {'Di':1.0, 'De':1.0, 'z':1.0,
                'ki_init':init_conds['K_i'],
                'ke_init':init_conds['K_e'],			 
                'k_scaling': 1,
                'f_k_i':src_terms['f_K_i'],
                'f_k_e':src_terms['f_K_e'],
                'J_k_e':bndry_terms['J_K_e'],
                'phi_i_e':exact_sols['phi_i_e'],
                'phi_e_e':exact_sols['phi_e_e'],
                'f_phi_i':src_terms['f_phi_i'],
                'f_phi_e':src_terms['f_phi_e'],
                'f_g_M':src_terms['f_g_M'],
                'f_I_M':src_terms['f_I_M'],
                'name':'K'}

        self.Cl = {'Di':1.0, 'De':1.0, 'z':-1.0,
                'ki_init':init_conds['Cl_i'],
                'ke_init':init_conds['Cl_e'],		
                'k_scaling': 1,	  
                'f_k_i':src_terms['f_Cl_i'],
                'f_k_e':src_terms['f_Cl_e'],
                'J_k_e':bndry_terms['J_Cl_e'],
                'phi_i_e':exact_sols['phi_i_e'],
                'phi_e_e':exact_sols['phi_e_e'],
                'f_phi_i':src_terms['f_phi_i'],
                'f_phi_e':src_terms['f_phi_e'],
                'f_g_M':src_terms['f_g_M'],
                'f_I_M':src_terms['f_I_M'],
                'name':'Cl'}

        # create ion list
        self.ion_list = [self.Na, self.K, self.Cl]

    def print_conservation(self):

        # Define measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        Na_i = self.wh[0].sub(0)
        K_i  = self.wh[0].sub(1)
        Cl_i = self.wh[0].sub(2)
        
        Na_e = self.wh[1].sub(0)
        K_e  = self.wh[1].sub(1)
        Cl_e = self.wh[1].sub(2)

        Na_tot = dfx.fem.assemble_scalar(dfx.fem.form(Na_i*dxi + Na_e*dxe, jit_options=self.jit_parameters))
        K_tot  = dfx.fem.assemble_scalar(dfx.fem.form(K_i *dxi + K_e *dxe, jit_options=self.jit_parameters))
        Cl_tot = dfx.fem.assemble_scalar(dfx.fem.form(Cl_i*dxi + Cl_e*dxe, jit_options=self.jit_parameters))

        
        print("Total Na+ concentration: ", self.comm.allreduce(Na_tot, op=MPI.SUM))
        print("Total K+  concentration: ", self.comm.allreduce(K_tot,  op=MPI.SUM))
        print("Total Cl- concentration: ", self.comm.allreduce(Cl_tot, op=MPI.SUM))

    def print_info(self):

        # define measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        Na_i  = self.wh[0].sub(0)
        K_i   = self.wh[0].sub(1)
        Cl_i  = self.wh[0].sub(2)		

        Na_e  = self.wh[1].sub(0)
        K_e   = self.wh[1].sub(1)
        Cl_e  = self.wh[1].sub(2)		

        Na_tot = dfx.fem.assemble_scalar(dfx.fem.form(Na_i*dxi)) + dfx.fem.assemble_scalar(dfx.fem.form(Na_e*dxe)) 
        K_tot  = dfx.fem.assemble_scalar(dfx.fem.form(K_i*dxi))  + dfx.fem.assemble_scalar(dfx.fem.form(K_e*dxe)) 
        Cl_tot = dfx.fem.assemble_scalar(dfx.fem.form(Cl_i*dxi)) + dfx.fem.assemble_scalar(dfx.fem.form(Cl_e*dxe)) 

        Na_tot = self.comm.allreduce(Na_tot, op=MPI.SUM)
        K_tot  = self.comm.allreduce(K_tot,  op=MPI.SUM)
        Cl_tot = self.comm.allreduce(Cl_tot, op=MPI.SUM)
        
        print("Na tot:", Na_tot)
        print("K  tot:",  K_tot)
        print("Cl tot:", Cl_tot)

    def print_errors(self):
        
        # Get the exact solutions
        if self.dim == 2:
            _, exact_sols, _, _ = self.M.get_MMS_terms_KNPEMI_2D(self.t.value)
        elif self.dim == 3:
            _, exact_sols, _, _ = self.M.get_MMS_terms_KNPEMI_3D(self.t.value)

        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        Na_i  = self.wh[0].sub(0)
        K_i   = self.wh[0].sub(1)
        Cl_i  = self.wh[0].sub(2)
        phi_i = self.wh[0].sub(3)

        Na_e  = self.wh[1].sub(0)
        K_e   = self.wh[1].sub(1)
        Cl_e  = self.wh[1].sub(2)
        phi_e = self.wh[1].sub(3)
        
        err_Na_i  = inner(Na_i  - exact_sols['Na_i_e'], Na_i   - exact_sols['Na_i_e'] ) * dxi
        err_K_i   = inner(K_i   - exact_sols['K_i_e'] , K_i    - exact_sols['K_i_e']  ) * dxi
        err_Cl_i  = inner(Cl_i  - exact_sols['Cl_i_e'], Cl_i   - exact_sols['Cl_i_e'] ) * dxi
        err_phi_i = inner(phi_i - exact_sols['phi_i_e'], phi_i - exact_sols['phi_i_e']) * dxi

        err_Na_e  = inner(Na_e  - exact_sols['Na_e_e'], Na_e   - exact_sols['Na_e_e'] ) * dxe
        err_K_e   = inner(K_e   - exact_sols['K_e_e'] , K_e    - exact_sols['K_e_e']  ) * dxe
        err_Cl_e  = inner(Cl_e  - exact_sols['Cl_e_e'], Cl_e   - exact_sols['Cl_e_e'] ) * dxe
        err_phi_e = inner(phi_e - exact_sols['phi_e_e'], phi_e - exact_sols['phi_e_e']) * dxe

        # calculate local L2 error norms
        L2_err_Na_i  = dfx.fem.assemble_scalar(dfx.fem.form(err_Na_i ))
        L2_err_K_i   = dfx.fem.assemble_scalar(dfx.fem.form(err_K_i  ))
        L2_err_Cl_i  = dfx.fem.assemble_scalar(dfx.fem.form(err_Cl_i ))
        L2_err_phi_i = dfx.fem.assemble_scalar(dfx.fem.form(err_phi_i))

        L2_err_Na_e  = dfx.fem.assemble_scalar(dfx.fem.form(err_Na_e ))
        L2_err_K_e   = dfx.fem.assemble_scalar(dfx.fem.form(err_K_e  ))
        L2_err_Cl_e  = dfx.fem.assemble_scalar(dfx.fem.form(err_Cl_e ))
        L2_err_phi_e = dfx.fem.assemble_scalar(dfx.fem.form(err_phi_e))

        # call allreduce and take square root to get global errors
        comm = self.comm # MPI communicator
        L2_err_Na_i  = np.sqrt(comm.allreduce(L2_err_Na_i,  op=MPI.SUM))
        L2_err_K_i   = np.sqrt(comm.allreduce(L2_err_K_i,   op=MPI.SUM))
        L2_err_Cl_i  = np.sqrt(comm.allreduce(L2_err_Cl_i,  op=MPI.SUM))
        L2_err_phi_i = np.sqrt(comm.allreduce(L2_err_phi_i, op=MPI.SUM))

        L2_err_Na_e  = np.sqrt(comm.allreduce(L2_err_Na_e,   op=MPI.SUM))
        L2_err_K_e   = np.sqrt(comm.allreduce(L2_err_K_e,    op=MPI.SUM))
        L2_err_Cl_e  = np.sqrt(comm.allreduce(L2_err_Cl_e,   op=MPI.SUM))
        L2_err_phi_e = np.sqrt(comm.allreduce(L2_err_phi_e,  op=MPI.SUM))

        # Print the errors
        print('#-------------- ERRORS --------------#')
        print('L2 Na_i  error:', L2_err_Na_i)
        print('L2 Na_e  error:', L2_err_Na_e)
        print('L2 K_i   error:', L2_err_K_i)
        print('L2 K_e   error:', L2_err_K_e)
        print('L2 Cl_i  error:', L2_err_Cl_i)
        print('L2 Cl_e  error:', L2_err_Cl_e)
        print('L2 phi_i error:', L2_err_phi_i)
        print('L2 phi_e error:', L2_err_phi_e)

        self.errors = [L2_err_Na_i, L2_err_Na_e, L2_err_K_i, L2_err_K_e, L2_err_Cl_i, L2_err_Cl_e, L2_err_phi_i, L2_err_phi_e]        

    ### Default class variables ###
    
    # Physical parameters
    C_M = 0.02                       # capacitance (F)
    T   = 300                        # temperature (K)
    F   = 96485                      # Faraday's constant (C/mol)
    R   = 8.314                      # Gas constant (J/(K*mol))
    psi = R*T/F                      # recurring variable
    
    g_Na_bar  = 1200                 # Na max conductivity (S/m**2)
    g_K_bar   = 360                  # K max conductivity (S/m**2)    
    g_Na_leak = 2.0*0.5              # Na leak conductivity (S/m**2) (Constant)
    g_K_leak  = 8.0*0.5              # K leak conductivity (S/m**2)
    g_Cl_leak = 0.0                  # Cl leak conductivity (S/m**2) (Constant)
    a_syn     = 0.002                # synaptic time constant (s)
    g_syn_bar = 40                   # synaptic conductivity (S/m**2)
    D_Na = 1.33e-9                   # diffusion coefficients Na (m/s^2) (Constant)
    D_K  = 1.96e-9                   # diffusion coefficients K (m/s^2) (Constant)
    D_Cl = 2.03e-9                   # diffusion coefficients Cl (m/s^2) (Constant)
    V_rest  = -0.065                 # resting membrane potential (V)

    # Potassium buffering params
    rho_pump = 1.115e-6			     # maximum pump rate (mol/m**2 s)
    P_Nai = 10                       # [Na+]i threshold for Na+/K+ pump (mol/m^3)
    P_Ke  = 1.5                      # [K+]e  threshold for Na+/K+ pump (mol/m^3)
    k_dec = 2.9e-8				     # Decay factor for [K+]e (m/s)

    # Initial conditions
    phi_e_init = 0         # external potential (V) (Constant)
    phi_i_init = -0.06774  # internal potential (V) just for visualization (Constant)
    phi_M_init = -0.06774  # membrane potential (V)	 (Constant)
    Na_i_init  = 12        # intracellular Na concentration (mol/m^3) (Constant)
    Na_e_init  = 100       # extracellular Na concentration (mol/m^3) (Constant)
    K_i_init   = 125       # intracellular K  concentration (mol/m^3) (Constant)
    K_e_init   = 4         # extracellular K  concentration (mol/m^3) (Constant)
    Cl_i_init  = 137       # intracellular Cl concentration (mol/m^3) (Constant)
    Cl_e_init  = 104       # extracellular Cl concentration (mol/m^3) (Constant)

    # Initial values of gating variables
    n_init_val = 0.27622914792
    m_init_val = 0.03791834627
    h_init_val = 0.68848921811

    # Source terms
    Na_e_f = 0.0
    Na_i_f = 0.0
    K_e_f  = 0.0
    K_i_f  = 0.0
    Cl_e_f = 0.0
    Cl_i_f = 0.0

    # Ion dictionaries and list
    Na = {'g_leak':g_Na_leak, 'Di':D_Na, 'De':D_Na, 'ki_init':Na_i_init, 'ke_init':Na_e_init, 'z':1.0,  'f_e': Na_e_f, 'f_i':Na_i_f, 'name':'Na', 'rho_p': 3*rho_pump}
    K  = {'g_leak':g_K_leak,  'Di':D_K,  'De':D_K,  'ki_init':K_i_init,  'ke_init':K_e_init,  'z':1.0,  'f_e': K_e_f,  'f_i':K_i_f,  'name':'K' , 'rho_p':-2*rho_pump}
    Cl = {'g_leak':g_Cl_leak, 'Di':D_Cl, 'De':D_Cl, 'ki_init':Cl_i_init, 'ke_init':Cl_e_init, 'z':-1.0, 'f_e': Cl_e_f, 'f_i':Cl_i_f, 'name':'Cl', 'rho_p':0.0}
    ion_list = [Na, K, Cl]
    N_ions   = len(ion_list) 

    # Mesh unit conversion factor
    mesh_conversion_factor = 1

    # Finite element polynomial order 
    fem_order = 1
    
    # Verification flag
    MMS_test        = False

    # Boundary condition type (only on phi, the electric potential)
    dirichlet_bcs   = False