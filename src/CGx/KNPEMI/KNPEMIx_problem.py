import ufl
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy   as np
import dolfinx as dfx
from dolfinx.fem import Constant

from ufl      import grad, inner, dot
from mpi4py   import MPI
from petsc4py import PETSc
from CGx.utils.mixed_dim_problem import MixedDimensionalProblem
from CGx.utils.setup_mms import ExactSolutionsKNPEMI
import basix.ufl

print = PETSc.Sys.Print # Automatically flushes output to stream in parallel

class ProblemKNPEMI(MixedDimensionalProblem):

    def init(self):
        """ Constructor. """

        if self.MMS_test: self.setup_MMS_params() # Perform numerical verification
        
    def setup_spaces(self):

        print("Setting up function spaces ...")

        # Define elements
        P = basix.ufl.element("Lagrange", self.mesh.basix_cell(), self.fem_order)

        # Ion concentrations for each ion + electric potential
        element_list = [P] * (self.N_ions + 1)

        self.V = dfx.fem.functionspace(self.mesh, basix.ufl.mixed_element(element_list))

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

        # Setup checkpoint output files for concentrations
        self.u_out_i = []
        self.u_out_e = []
        for idx, ion in enumerate(self.ion_list):
            intra_func = self.u_p[0].sub(idx).collapse()
            intra_func.name = f"{ion["name"]}_i"
            self.u_out_i.append(intra_func)
            extra_func = self.u_p[1].sub(idx).collapse()
            extra_func.name = f"{ion["name"]}_e"
            self.u_out_e.append(extra_func)

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
        
        Wi = self.W[0]
        We = self.W[1]

        # Add Dirichlet boundary conditions on exterior boundary
        bcs = []

        if self.dirichlet_bcs:
            
            facets_boundary = self.boundaries.find(self.boundary_tag[0])

            # First round in for-loop is for intracellular variables
            ion_suffix = 'i'
            init_phi = self.phi_m_init
            for W in [Wi, We]:
                # BCs for concentrations
                for idx, ion in enumerate(self.ion_list):

                    W_ion, _ = W.sub(idx).collapse()
                    func = dfx.fem.Function(W_ion)
                    if self.MMS_test:
                        func.interpolate(ion[f'k{ion_suffix}_init'])
                    else:
                        func.x.array[:] = ion[f'k{ion_suffix}_init']

                    dofs = dfx.fem.locate_dofs_topological((W.sub(idx), W_ion), self.boundaries.dim, facets_boundary)
                    bcs.append(dfx.fem.dirichletbc(func, dofs, W.sub(idx)))

                # Electric potential in extracellular space
                W_phi, _ = W.sub(self.N_ions).collapse()
                func = dfx.fem.Function(W_phi)

                if self.MMS_test:
                    func.interpolate(dfx.fem.Expression(
                                        ion['ke_init'],
                                        W_ion.element.interpolation_points()
                                        )
                                    )
                else:
                    func.x.array[:] = init_phi
                    
                dofs = dfx.fem.locate_dofs_topological((W.sub(self.N_ions), W_phi), self.boundaries.dim, facets_boundary)
                bcs.append(dfx.fem.dirichletbc(func, dofs, W.sub(self.N_ions)))

                # Next round in for-loop is for extracellular variables
                ion_suffix = 'e'
                init_phi = 0.0

        self.bcs = bcs

    def setup_source_terms(self):
        """ Initialize source term functions. """

        Ve_K,  _ = self.W[1].sub(1).collapse()
        Ve_Cl, _ = self.W[1].sub(2).collapse()
        f_e_K  = dfx.fem.Function(Ve_K)
        f_e_Cl = dfx.fem.Function(Ve_Cl)
        
        injection_dofs_K  = dfx.fem.locate_dofs_topological(Ve_K,  self.subdomains.dim, self.injection_cells)
        injection_dofs_Cl = dfx.fem.locate_dofs_topological(Ve_Cl, self.subdomains.dim, self.injection_cells)
        vol = self.injection_volume
        I = 5e-9 # Ion injection current of 5 nA
        mol_rate = I / (1*self.F) # [mol/s]
        src_term = mol_rate / vol # [mol/L/s = M/s]
        f_e_K.x.array[injection_dofs_K]   = src_term
        f_e_Cl.x.array[injection_dofs_Cl] = src_term

        self.ion_list[1]['f_e'] = f_e_K
        self.ion_list[2]['f_e'] = f_e_Cl

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

    def set_initial_conditions(self):

        # Set initial electric potentials
        phi_space, _ = self.V.sub(self.N_ions).collapse()
        self.phi_m_prev = dfx.fem.Function(phi_space)
        self.phi_m_prev.name = "phi_m"
        if self.MMS_test:
            self.phi_m_prev.interpolate(
                                    dfx.fem.Expression(
                                        self.phi_m_init,
                                        phi_space.element.interpolation_points()
                                        )
                                    )
            # Set intra- and extracellular potentials just for visualization   
            ui_space, _ = ui_p.sub(self.N_ions).function_space.collapse()
            ue_space, _ = ue_p.sub(self.N_ions).function_space.collapse()
            ui_p.sub(self.N_ions).interpolate(
                                    dfx.fem.Expression(
                                        self.phi_i_init,
                                        ui_space.element.interpolation_points()
                                        )
                                    )
            ue_p.sub(self.N_ions).interpolate(
                                    dfx.fem.Expression(
                                        self.phi_e_init,
                                        ue_space.element.interpolation_points()
                                        )
                                    )
        else:
            if self.glia_tags is None:
                # Only neuronal cells
                self.phi_m_prev.x.array[:] = self.phi_m_init.value
            else:
                # Both neuronal and glial cells
                neuron_dofs = dfx.fem.locate_dofs_topological(phi_space, self.subdomains.dim, self.neuron_cells)
                glia_dofs = dfx.fem.locate_dofs_topological(phi_space, self.subdomains.dim, self.glia_cells)

                self.phi_m_prev.x.array[neuron_dofs] = self.phi_m_n_init.value
                self.phi_m_prev.x.array[glia_dofs] = self.phi_m_g_init.value

        # Set initial concentrations
        # Solutions at previous timestep
        ui_p = self.u_p[0]
        ue_p = self.u_p[1]

        # Initialize parts of variational formulation
        for idx, ion in enumerate(self.ion_list):
            # Set initial value of intra- and extracellular ion concentrations
            if self.MMS_test:
                ui_space, _ = ui_p.sub(idx).function_space.collapse()
                ue_space, _ = ue_p.sub(idx).function_space.collapse()
                ui_p.sub(idx).interpolate(
                                    dfx.fem.Expression(
                                        ion['ki_init'],
                                        ui_space.element.interpolation_points()
                                        )
                                    )
                ue_p.sub(idx).interpolate(
                                    dfx.fem.Expression(
                                        ion['ke_init'],
                                        ue_space.element.interpolation_points()
                                        )
                                    )
            else:
                if self.glia_tags is None:
                    # Get dof mapping between subspace and parent space
                    _, sub_to_parent_i = ui_p.sub(idx).function_space.collapse()
                    _, sub_to_parent_e = ue_p.sub(idx).function_space.collapse()

                    # Set the array values at the subspace dofs 
                    ui_p.sub(idx).x.array[sub_to_parent_i] = ion['ki_init'].value
                    ue_p.sub(idx).x.array[sub_to_parent_e] = ion['ke_init'].value
                else:
                    # Get dof mapping between subspace and parent space
                    W_ion_i, sub_to_parent_i = ui_p.sub(idx).function_space.collapse()
                    neuron_dofs = dfx.fem.locate_dofs_topological((self.W[0].sub(idx), W_ion_i), self.subdomains.dim, self.neuron_cells)
                    glia_dofs   = dfx.fem.locate_dofs_topological((self.W[0].sub(idx), W_ion_i), self.subdomains.dim, self.glia_cells)
                    _, sub_to_parent_e = ue_p.sub(idx).function_space.collapse()

                    # Set the array values at the subspace dofs 
                    ui_p.sub(idx).x.array[np.intersect1d(sub_to_parent_i, neuron_dofs[0])] = ion['ki_init_n'].value
                    ui_p.sub(idx).x.array[np.intersect1d(sub_to_parent_i, glia_dofs[0])]   = ion['ki_init_g'].value
                    ue_p.sub(idx).x.array[sub_to_parent_e] = ion['ke_init'].value
        
        print("Initial conditions set.")
            
    def setup_variational_form(self):

        print("Setting up variational form ...")
        
        # Check that ionic models have been initialized
        if len(self.ionic_models)==0 and self.comm.rank==0:
            raise RuntimeError('\nNo ionic model(s) specified.\nCall init_ionic_model() to provide ionic models.\n')

        # Aliases
        dt  = self.dt
        F   = self.F
        psi = self.psi
        C_M = self.C_M

        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags)

        # For the MMS test various gamma faces get different tags
        if self.MMS_test:
            # Create boundary integral measure
            ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
            
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
        
                    # Add stimulus current if the current cell membrane belongs to
                    # a cell that is stimulated
                    if gamma_tag in self.stimulus_tags:
                        if ion['name']=='Na' and model.__str__()=='Hodgkin-Huxley':
                            stim = model._add_stimulus(idx)
                            # self.stim, self.stim_expr = model._add_stimulus(idx)
                            ion['I_ch'][gamma_tag] += stim

                    # Add contribution to total channel current
                    I_ch[gamma_tag] += ion['I_ch'][gamma_tag]
        
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
                L0 -= (dt*I_ch_k[gamma_tag] - alpha_i('-')*C_M*self.phi_m_prev) / (F*z) * vki('-') * dS(gamma_tag)
                L1 += (dt*I_ch_k[gamma_tag] - alpha_e('-')*C_M*self.phi_m_prev) / (F*z) * vke('-') * dS(gamma_tag)

            # Add contributions to total current flux
            J_phi_i += z*Ji
            J_phi_e += z*Je

            # Source terms
            L0 += dt * inner(ion['f_i'], vki) * dxi
            L1 += dt * inner(ion['f_e'], vke) * dxe

            if self.MMS_test:
                # Define outward normal on exterior boundary (\partial\Omega)
                n_outer = ufl.FacetNormal(self.mesh)

                # Concentrations source terms
                L0 += dt * inner(ion['f_k_i'], vki) * dxi # Equation for k_i
                L1 += dt * inner(ion['f_k_e'], vke) * dxe # Equation for k_e

                # Enforcing correction for I_m
                L0 += dt/(F*z) * alpha_i('-') * inner(ion['f_I_M'], vki('-')) * dS(self.gamma_tags)
                L1 -= dt/(F*z) * alpha_e('-') * inner(ion['f_I_M'], vke('-')) * dS(self.gamma_tags)
            
                # Enforcing correction for I_m, assuming gM_k = gM / N_ions
                L1 -= dt/(F*z) * alpha_e('-')*inner(ion['f_g_M'], vke('-'))*dS(self.gamma_tags)

                # Exterior boundary terms (zero in "physical problem")
                L1 -=  dt * inner(dot(ion['J_k_e'], n_outer), vke) * ds # Equation for k_e
                L1 += F*z * inner(dot(ion['J_k_e'], n_outer), vphi_e) * ds # Equation for phi_e
        
        # Weak form - equation for phi_i
        a00 -= inner(J_phi_i, grad(vphi_i)) * dxi - (C_M/(F*dt)) * inner(phi_i('-'), vphi_i('-')) * dS
        a01 -= (C_M/(F*dt)) * inner(phi_e('-'), vphi_i('-')) * dS
        
        # Weak form - equation for phi_e
        a11 -= inner(J_phi_e, grad(vphi_e)) * dxe - (C_M/(F*dt)) * inner(phi_e('-'), vphi_e('-')) * dS
        a10 -= (C_M/(F*dt)) * inner(phi_i('-'), vphi_e('-')) * dS 
        
        for gamma_tag in self.gamma_tags:
            L0  -= (1/F) * (I_ch[gamma_tag] - C_M*self.phi_m_prev/dt) * vphi_i('-') * dS(gamma_tag)
            L1  += (1/F) * (I_ch[gamma_tag] - C_M*self.phi_m_prev/dt) * vphi_e('-') * dS(gamma_tag)
        
        if self.MMS_test:
            # Phi source terms
            L0 -= inner(ion['f_phi_i'], vphi_i) * dxi # Equation for phi_i
            L1 -= inner(ion['f_phi_e'], vphi_e) * dxe # Equation for phi_e

            # Enforcing correction for I_m
            
            L0 += inner(self.src_terms['f_phi_m'], vphi_i('-')) * dS(self.gamma_tags)
            L1 -= inner(self.src_terms['f_phi_m'], vphi_e('-')) * dS(self.gamma_tags)

            L1 -= inner(self.src_terms['f_gamma'], vphi_e('-')) * dS(self.gamma_tags)
              

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

            # Add contribution to total current flux
            if use_block_jacobi:
                Ji = - Di*z/psi * ki_prev*grad(phi_i)
                Je = - De*z/psi * ke_prev*grad(phi_e)
            else:
                Ji = - Di*grad(ki) - Di*z/psi * ki_prev*grad(phi_i)
                Je = - De*grad(ke) - De*z/psi * ke_prev*grad(phi_e)
            
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
    
        assert np.allclose([self.C_M, self.R, self.F], [1.0]*3) 
        self.psi = 1.0

        self.M = ExactSolutionsKNPEMI(self.mesh, self.t)
        
        if self.mesh.geometry.dim == 2:
            exact_sols, src_terms = self.M.get_mms_terms()
        elif self.mesh.geometry.dim == 3:
            src_terms, exact_sols, bndry_terms = self.M.get_MMS_terms_KNPEMI_3D(0.0)
        
        self.exact_sols = exact_sols
        self.src_terms  = src_terms

        # initial values
        self.phi_i_init = exact_sols["phi_i_init"] # internal potential (V) just for visualization
        self.phi_e_init = exact_sols["phi_e_init"] # external potential (V) just for visualization
        self.phi_m_init = self.phi_i_init - self.phi_e_init # membrane potential (V)

        # create ions
        self.Na = {'Di' : dfx.fem.Constant(self.mesh, 1.0),
                   'De' : dfx.fem.Constant(self.mesh, 1.0),
                   'z'  : dfx.fem.Constant(self.mesh, 1.0),
                   'ki_init':exact_sols['Na_i'],
                   'ke_init':exact_sols['Na_e'],		
                   'k_scaling': dfx.fem.Constant(self.mesh, 1.0),	  
                   'f_k_i':src_terms['f_Na_i'],
                   'f_k_e':src_terms['f_Na_e'],
                   'J_k_e':src_terms['J_Na_e'],
                   'phi_i_e':exact_sols['phi_i'],
                   'phi_e_e':exact_sols['phi_e'],
                   'f_phi_i':src_terms['f_phi_i'],
                   'f_phi_e':src_terms['f_phi_e'],
                   'f_g_M':src_terms['f_gamma'],
                   'f_I_M':src_terms['f_phi_Na'],
                   'name':'Na',
                   'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                   'f_e' : dfx.fem.Constant(self.mesh, 0.0)}

        self.K = {'Di' : dfx.fem.Constant(self.mesh, 1.0),
                  'De' : dfx.fem.Constant(self.mesh, 1.0),
                  'z'  : dfx.fem.Constant(self.mesh, 1.0),
                  'ki_init':exact_sols['K_i'],
                  'ke_init':exact_sols['K_e'],			 
                  'k_scaling': dfx.fem.Constant(self.mesh, 1.0),
                  'f_k_i':src_terms['f_K_i'],
                  'f_k_e':src_terms['f_K_e'],
                  'J_k_e':src_terms['J_K_e'],
                  'phi_i_e':exact_sols['phi_i'],
                  'phi_e_e':exact_sols['phi_e'],
                  'f_phi_i':src_terms['f_phi_i'],
                  'f_phi_e':src_terms['f_phi_e'],
                  'f_g_M':src_terms['f_gamma'],
                  'f_I_M':src_terms['f_phi_K'],
                  'name':'K',
                  'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                  'f_e' : dfx.fem.Constant(self.mesh, 0.0)}

        self.Cl = {'Di' : dfx.fem.Constant(self.mesh, 1.0),
                   'De' : dfx.fem.Constant(self.mesh, 1.0),
                   'z'  : dfx.fem.Constant(self.mesh, 1.0),
                   'ki_init':exact_sols['Cl_i'],
                   'ke_init':exact_sols['Cl_e'],		
                   'k_scaling': dfx.fem.Constant(self.mesh, 1.0),	  
                   'f_k_i':src_terms['f_Cl_i'],
                   'f_k_e':src_terms['f_Cl_e'],
                   'J_k_e':src_terms['J_Cl_e'],
                   'phi_i_e':exact_sols['phi_i'],
                   'phi_e_e':exact_sols['phi_e'],
                   'f_phi_i':src_terms['f_phi_i'],
                   'f_phi_e':src_terms['f_phi_e'],
                   'f_g_M':src_terms['f_gamma'],
                   'f_I_M':src_terms['f_phi_Cl'],
                   'name':'Cl',
                   'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                   'f_e' : dfx.fem.Constant(self.mesh, 0.0)}

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
        
        exact_sols = self.exact_sols
        
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
        
        err_Na_i  = inner(Na_i  - exact_sols['Na_i'], Na_i   - exact_sols['Na_i'] ) * dxi
        err_K_i   = inner(K_i   - exact_sols['K_i'] , K_i    - exact_sols['K_i']  ) * dxi
        err_Cl_i  = inner(Cl_i  - exact_sols['Cl_i'], Cl_i   - exact_sols['Cl_i'] ) * dxi
        err_phi_i = inner(phi_i - exact_sols['phi_i'], phi_i - exact_sols['phi_i']) * dxi

        err_Na_e  = inner(Na_e  - exact_sols['Na_e'], Na_e   - exact_sols['Na_e'] ) * dxe
        err_K_e   = inner(K_e   - exact_sols['K_e'] , K_e    - exact_sols['K_e']  ) * dxe
        err_Cl_e  = inner(Cl_e  - exact_sols['Cl_e'], Cl_e   - exact_sols['Cl_e'] ) * dxe
        err_phi_e = inner(phi_e - exact_sols['phi_e'], phi_e - exact_sols['phi_e']) * dxe

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

    def setup_constants(self):
        """ Set default class variables. """
        # Physical parameters
        self.C_M = Constant(self.mesh, dfx.default_scalar_type(0.02))  # Capacitance (F)
        self.T   = Constant(self.mesh, dfx.default_scalar_type(300))   # Temperature (K)
        self.F   = Constant(self.mesh, dfx.default_scalar_type(96485)) # Faraday's constant (C/mol)
        self.R   = Constant(self.mesh, dfx.default_scalar_type(8.314)) # Gas constant (J/(K*mol))
        self.psi = Constant(self.mesh,          
                        dfx.default_scalar_type(    
                            self.R.value*self.T.value/self.F.value     # Recurring variable
                            )
                        ) 
    
        self.g_Na_bar  = Constant(self.mesh, dfx.default_scalar_type(1200)) # Na max conductivity (S/m**2)
        self.g_K_bar   = Constant(self.mesh, dfx.default_scalar_type(360))  # K max conductivity (S/m**2)    
        self.g_Na_leak   = Constant(self.mesh, dfx.default_scalar_type(1.0)) # Na leak conductivity (S/m**2) (Constant)
        self.g_Na_leak_g = Constant(self.mesh, dfx.default_scalar_type(1.0)) # Na leak conductivity (S/m**2) (Constant)
        self.g_K_leak    = Constant(self.mesh, dfx.default_scalar_type(4.0)) # K leak conductivity (S/m**2)
        self.g_K_leak_g  = Constant(self.mesh, dfx.default_scalar_type(16.96)) # K leak conductivity (S/m**2)
        self.g_Cl_leak   = Constant(self.mesh, dfx.default_scalar_type(0.25)) # Cl leak conductivity (S/m**2) (Constant)
        self.g_Cl_leak_g = Constant(self.mesh, dfx.default_scalar_type(0.50)) # Cl leak conductivity (S/m**2) (Constant)
        self.a_syn     = Constant(self.mesh, dfx.default_scalar_type(0.002)) # Synaptic time constant (s)
        self.g_syn_bar = Constant(self.mesh, dfx.default_scalar_type(40)) # Synaptic conductivity (S/m**2)
        self.D_Na = Constant(self.mesh, dfx.default_scalar_type(1.33e-9)) # Diffusion coefficients Na (m/s^2) (Constant)
        self.D_K  = Constant(self.mesh, dfx.default_scalar_type(1.96e-9)) # Diffusion coefficients K (m/s^2) (Constant)
        self.D_Cl = Constant(self.mesh, dfx.default_scalar_type(2.03e-9)) # diffusion coefficients Cl (m/s^2) (Constant)
        self.phi_rest  = Constant(self.mesh, dfx.default_scalar_type(-0.065)) # Resting membrane potential (V)

        # Potassium buffering params
        self.rho_pump = Constant(self.mesh, dfx.default_scalar_type(1.115e-6)) # maximum pump rate (mol/m**2 s)
        self.P_Nai = Constant(self.mesh, dfx.default_scalar_type(10))          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
        self.P_Ke  = Constant(self.mesh, dfx.default_scalar_type(1.5))         # [K+]e  threshold for Na+/K+ pump (mol/m^3)
        self.k_dec = Constant(self.mesh, dfx.default_scalar_type(2.9e-8))	  # Decay factor for [K+]e (m/s)

        # Initial conditions
        self.phi_m_init = Constant(self.mesh, dfx.default_scalar_type(-0.067))  # Membrane potential, neuronal (V) 
        self.phi_m_init_g = Constant(self.mesh, dfx.default_scalar_type(-0.085))  # Membrane potential, glial (V) 
        self.Na_i_init  = Constant(self.mesh, dfx.default_scalar_type(12))        # Intracellular Na concentration (mol/m^3) (Constant)
        self.Na_e_init  = Constant(self.mesh, dfx.default_scalar_type(100))       # Extracellular Na concentration (mol/m^3) (Constant)
        self.K_i_init   = Constant(self.mesh, dfx.default_scalar_type(125))       # Intracellular K  concentration (mol/m^3) (Constant)
        self.K_e_init   = Constant(self.mesh, dfx.default_scalar_type(4))         # Extracellular K  concentration (mol/m^3) (Constant)
        self.Cl_i_init  = Constant(self.mesh, dfx.default_scalar_type(137))       # Intracellular Cl concentration (mol/m^3) (Constant)
        self.Cl_e_init  = Constant(self.mesh, dfx.default_scalar_type(104))       # Extracellular Cl concentration (mol/m^3) (Constant)

        # Initial values of gating variables
        self.n_init = Constant(self.mesh, dfx.default_scalar_type(0.276))
        self.m_init = Constant(self.mesh, dfx.default_scalar_type(0.00379))
        self.h_init = Constant(self.mesh, dfx.default_scalar_type(0.688))

        # Source terms
        self.Na_e_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Na_i_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.K_e_f  = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.K_i_f  = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Cl_e_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Cl_i_f = Constant(self.mesh, dfx.default_scalar_type(0.0))

        # Ion dictionaries and list
        self.Na = {'g_leak':self.g_Na_leak, 'Di':self.D_Na, 'De':self.D_Na, 'ki_init':self.Na_i_init, 'ke_init':self.Na_e_init, 'z':Constant(self.mesh, 1.0),  'f_e': self.Na_e_f, 'f_i':self.Na_i_f, 'name':'Na', 'rho_p': Constant(self.mesh, dfx.default_scalar_type(3*self.rho_pump.value))}
        self.K  = {'g_leak':self.g_K_leak,  'Di':self.D_K,  'De':self.D_K,  'ki_init':self.K_i_init,  'ke_init':self.K_e_init,  'z':Constant(self.mesh, 1.0),  'f_e': self.K_e_f,  'f_i':self.K_i_f,  'name':'K' , 'rho_p': Constant(self.mesh, dfx.default_scalar_type(-2*self.rho_pump.value))}
        self.Cl = {'g_leak':self.g_Cl_leak, 'Di':self.D_Cl, 'De':self.D_Cl, 'ki_init':self.Cl_i_init, 'ke_init':self.Cl_e_init, 'z':Constant(self.mesh, -1.0), 'f_e': self.Cl_e_f, 'f_i':self.Cl_i_f, 'name':'Cl', 'rho_p': Constant(self.mesh, 0.0)}
        self.ion_list = [self.Na, self.K, self.Cl]
        self.N_ions   = len(self.ion_list) 

    # Class settings
    # Mesh unit conversion factor
    mesh_conversion_factor = 1

    # Finite element polynomial order 
    fem_order = 1
    
    # Verification flag
    MMS_test        = False

    # Boundary condition type 
    dirichlet_bcs   = False