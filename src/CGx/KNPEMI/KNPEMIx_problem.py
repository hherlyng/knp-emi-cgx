import ufl
import basix.ufl
import multiphenicsx
import multiphenicsx.fem

import numpy   as np
import dolfinx as dfx
from dolfinx.fem import Constant

from ufl      import grad, inner, dot
from mpi4py   import MPI
from petsc4py import PETSc
from CGx.utils.setup_mms import ExactSolutionsKNPEMI
from CGx.utils.mixed_dim_problem import MixedDimensionalProblem
from CGx.utils.membrane_ODE_systems import TwoCompartmentMembraneODESystem, ThreeCompartmentMembraneODESystem
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley

print = PETSc.Sys.Print # Automatically flushes output to stream in parallel

class ProblemKNPEMI(MixedDimensionalProblem):
    
    def init(self):
        """ Constructor. """

        if self.MMS_test: self.setup_MMS_params()

    def setup_spaces(self):

        print("Setting up function spaces ...")

        # Define number of variables: Ion concentrations + electric potential,
        # times two because of intra- and extracellular spaces
        self.num_variables = (self.N_ions + 1)
        num_variables_total = 2*self.num_variables

        # Define elements: Continuous Lagrange elements of order 'fem_order'
        P = basix.ufl.element(
                "Lagrange",
                self.mesh.basix_cell(),
                self.fem_order
            )

        # Create function spaces
        self.V = dfx.fem.functionspace(self.mesh, P) # Continuous Lagrange space 
        self.V_list = [self.V.clone() for _ in range(num_variables_total)] # List of each variable's function space
        self.V_list_ie = [self.V_list[:self.num_variables], self.V_list[self.num_variables:]] # Separated list with intra- and extracellular variables split
        self.W = ufl.MixedFunctionSpace(*self.V_list) # Mixed function space

        # Functions for storing the solutions
        self.wh = [[dfx.fem.Function(V) for V in self.V_list_ie[0]], [dfx.fem.Function(V) for V in self.V_list_ie[1]]]

        # Functions for solution at previous timestep
        self.u_p = [[dfx.fem.Function(V) for V in self.V_list_ie[0]], [dfx.fem.Function(V) for V in self.V_list_ie[1]]]

        # Setup checkpoint output files
        self.u_out_i = []
        self.u_out_e = []
        for idx, ion in enumerate(self.ion_list):
            intra_func = self.u_p[0][idx]
            intra_func.name = f"{ion['name']}_i"
            self.u_out_i.append(intra_func)
            extra_func = self.u_p[1][idx]
            extra_func.name = f"{ion['name']}_e"
            self.u_out_e.append(extra_func)
        phi_i = self.u_p[0][self.N_ions]
        phi_i.name = "phi_i"
        self.u_out_i.append(phi_i)
        phi_e = self.u_p[1][self.N_ions]
        phi_e.name = "phi_e"
        self.u_out_e.append(phi_e)


        ## Restrictions ##
        print("Creating mesh restrictions ...")
        
        # Get indices of the cells of the intra- and extracellular subdomains        
        if len(self.intra_tags) > 1:
            intra_indices = np.concatenate(([self.subdomains.find(tag) for tag in self.intra_tags]))
        else:
            intra_indices = self.subdomains.values==self.intra_tags[0]
        
        extra_indices = self.subdomains.values==self.extra_tag
        
        cells_intra = self.subdomains.indices[intra_indices]
        cells_extra = self.subdomains.indices[extra_indices]

        self.dofs_intra = dfx.fem.locate_dofs_topological(self.V, self.subdomains.dim, cells_intra)
        self.dofs_extra = dfx.fem.locate_dofs_topological(self.V, self.subdomains.dim, cells_extra)

        self.interior = multiphenicsx.fem.DofMapRestriction(self.V.dofmap, self.dofs_intra)
        self.exterior = multiphenicsx.fem.DofMapRestriction(self.V.dofmap, self.dofs_extra)

        # Get interior and exterior dofs
        self.restriction = [None] * num_variables_total
        self.restriction[:self.num_variables] = [self.interior] * self.num_variables
        self.restriction[self.num_variables:] = [self.exterior] * self.num_variables

    def setup_boundary_conditions(self):

        print('Setting up boundary conditions ...')
        
        Wi = self.V_list[:self.num_variables]
        We = self.V_list[self.num_variables:]

        # Add Dirichlet boundary conditions on exterior boundary
        bcs = []

        if self.dirichlet_bcs:
            facets_boundary = np.concatenate(([self.boundaries.find(tag) for tag in self.boundary_tags]))
        
            if self.MMS_test:
                # Ion concentrations in extracellular space
                for idx, ion in enumerate(self.ion_list):
                    W_ion = We[idx]
                    func = dfx.fem.Function(W_ion)
                    func.interpolate(
                        dfx.fem.Expression(
                            self.exact_sols[ion['name']+'_e'],
                            W_ion.element.interpolation_points()
                        )
                    )
                    
                    dofs = dfx.fem.locate_dofs_topological(W_ion, self.boundaries.dim, facets_boundary)
                    bcs.append(dfx.fem.dirichletbc(func, dofs))

                # Electric potential in extracellular space
                W_phi = We[self.N_ions]
                func = dfx.fem.Function(W_phi)
                func.interpolate(
                        dfx.fem.Expression(
                            self.exact_sols['phi_e'],
                            W_phi.element.interpolation_points()
                        )
                    )
                dofs = dfx.fem.locate_dofs_topological(W_phi, self.boundaries.dim, facets_boundary)
                bcs.append(dfx.fem.dirichletbc(func, dofs))

            else:
                # First round in for-loop is for intracellular variables
                ion_suffix = 'i'
                init_phi = self.phi_m_init.value
            
                for W in [Wi, We]:
                    # BCs for concentrations
                    for idx, ion in enumerate(self.ion_list):

                        W_ion = W[idx]
                        func = dfx.fem.Function(W_ion)
                        func.x.array[:] = ion[f'k{ion_suffix}_init'].value

                        dofs = dfx.fem.locate_dofs_topological(W_ion, self.boundaries.dim, facets_boundary)
                        bcs.append(dfx.fem.dirichletbc(func, dofs))

                    # Electric potential in extracellular space
                    W_phi = W[self.N_ions]
                    func = dfx.fem.Function(W_phi)
                    func.x.array[:] = init_phi
                        
                    dofs = dfx.fem.locate_dofs_topological(W_phi, self.boundaries.dim, facets_boundary)
                    bcs.append(dfx.fem.dirichletbc(func, dofs))

                    # Next round in for-loop is for extracellular variables
                    ion_suffix = 'e'
                    init_phi = 0.0

        self.bcs = bcs

    def setup_source_terms(self):
        """ Initialize source term functions. """

        Ve_K  = self.V_list_ie[1][1]
        Ve_Cl = self.V_list_ie[1][2]
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

    def set_initial_conditions(self):
        
        if self.MMS_test:
            print("Setting initial conditions for MMS test ...")
        elif self.find_initial_conditions:
            print("Solving ODE system to find steady-state initial conditions ...")
            comm: MPI.Comm = self.comm
            self.calculate_compartment_volumes_and_surface_areas()
            if self.glia_tags is None:
                membrane_odes = TwoCompartmentMembraneODESystem(self, plot_flags=[True, False, True], stimulus_flag=False)
                if comm.rank==0:
                    # Solve the membrane ODE system to find a steady-state
                    (phi_m_init_val, Na_i_init_val, Na_e_init_val, K_i_init_val, K_e_init_val,
                     Cl_i_init_val, Cl_e_init_val, n_init_val, m_init_val, h_init_val) = membrane_odes.solve_ode_system()
                else:
                    # Placeholders on non-root ranks
                    phi_m_init_val = None
                    Na_i_init_val = None
                    Na_e_init_val = None
                    K_i_init_val = None
                    K_e_init_val = None
                    Cl_i_init_val = None
                    Cl_e_init_val = None
                    n_init_val = None
                    m_init_val = None
                    h_init_val = None
                # Communicate initial values from root process
                phi_m_init_val = comm.bcast(phi_m_init_val, root=0)
                Na_i_init_val = comm.bcast(Na_i_init_val, root=0)
                Na_e_init_val = comm.bcast(Na_e_init_val, root=0)
                K_i_init_val = comm.bcast(K_i_init_val, root=0)
                K_e_init_val = comm.bcast(K_e_init_val, root=0)
                Cl_i_init_val = comm.bcast(Cl_i_init_val, root=0)
                Cl_e_init_val = comm.bcast(Cl_e_init_val, root=0)
                n_init_val = comm.bcast(n_init_val, root=0)
                m_init_val = comm.bcast(m_init_val, root=0)
                h_init_val = comm.bcast(h_init_val, root=0)
                # Update values of constants
                self.phi_m_init.value = phi_m_init_val
                self.Na_i_init.value = Na_i_init_val
                self.Na_e_init.value = Na_e_init_val
                self.K_i_init.value = K_i_init_val
                self.K_e_init.value = K_e_init_val
                self.Cl_i_init.value = Cl_i_init_val
                self.Cl_e_init.value = Cl_e_init_val
                self.n_init.value = n_init_val
                self.m_init.value = m_init_val
                self.h_init.value = h_init_val
            else:
                membrane_odes = ThreeCompartmentMembraneODESystem(self, plot_flags=[True, False, True], stimulus_flag=False)
                if comm.rank==0:
                    # Solve the membrane ODE system to find a steady-state
                    (phi_m_n_init_val, Na_i_n_init_val, Na_e_init_val, K_i_n_init_val, K_e_init_val,
                    Cl_i_n_init_val, Cl_e_init_val, phi_m_g_init_val, Na_i_g_init_val, K_i_g_init_val, Cl_i_g_init_val,
                    n_init_val, m_init_val, h_init_val) = membrane_odes.solve_ode_system()
                else:
                    # Placeholders on non-root ranks
                    phi_m_n_init_val = None
                    Na_i_n_init_val = None
                    Na_e_init_val = None
                    K_i_n_init_val = None
                    K_e_init_val = None
                    Cl_i_n_init_val = None
                    Cl_e_init_val = None
                    phi_m_g_init_val = None
                    Na_i_g_init_val = None
                    K_i_g_init_val = None
                    Cl_i_g_init_val = None
                    n_init_val = None
                    m_init_val = None
                    h_init_val = None

                # Communicate initial values from root process
                phi_m_n_init_val = comm.bcast(phi_m_n_init_val, root=0)
                Na_i_n_init_val = comm.bcast(Na_i_n_init_val, root=0)
                Na_e_init_val = comm.bcast(Na_e_init_val, root=0)
                K_i_n_init_val = comm.bcast(K_i_n_init_val, root=0)
                K_e_init_val = comm.bcast(K_e_init_val, root=0)
                Cl_i_n_init_val = comm.bcast(Cl_i_n_init_val, root=0)
                Cl_e_init_val = comm.bcast(Cl_e_init_val, root=0)
                phi_m_g_init_val = comm.bcast(phi_m_g_init_val, root=0)
                Na_i_g_init_val = comm.bcast(Na_i_g_init_val, root=0)
                K_i_g_init_val = comm.bcast(K_i_g_init_val, root=0)
                Cl_i_g_init_val = comm.bcast(Cl_i_g_init_val, root=0)
                n_init_val = comm.bcast(n_init_val, root=0)
                m_init_val = comm.bcast(m_init_val, root=0)
                h_init_val = comm.bcast(h_init_val, root=0)

                # Update values of constants
                self.phi_m_n_init.value = phi_m_n_init_val
                self.Na_i_n_init.value = Na_i_n_init_val
                self.Na_e_init.value = Na_e_init_val
                self.K_i_n_init.value = K_i_n_init_val
                self.K_e_init.value = K_e_init_val
                self.Cl_i_n_init.value = Cl_i_n_init_val
                self.Cl_e_init.value = Cl_e_init_val
                self.phi_m_g_init.value = phi_m_g_init_val
                self.Na_i_g_init.value = Na_i_g_init_val
                self.K_i_g_init.value = K_i_g_init_val
                self.Cl_i_g_init.value = Cl_i_g_init_val
                self.n_init.value = n_init_val
                self.m_init.value = m_init_val
                self.h_init.value = h_init_val
        else:
            print("Setting initial conditions from input file ...")
            if self.glia_tags is None:
                self.phi_m_init.value = self.initial_conditions['phi_m'] # Membrane potential
                self.Na_i_init.value = self.initial_conditions['Na_i'] # Intracellular Na+ concentration
                self.Na_e_init.value = self.initial_conditions['Na_e'] # Extracellular Na+ concentration
                self.K_i_init.value = self.initial_conditions['K_i'] # Intracellular K+ concentration
                self.K_e_init.value = self.initial_conditions['K_e'] # Extracellular K+ concentration
                self.Cl_i_init.value = self.initial_conditions['Cl_i'] # Intracellular Cl- concentration
                self.Cl_e_init.value = self.initial_conditions['Cl_e'] # Extracellular Cl- concentration
                self.n_init.value = self.initial_conditions['n'] # K+ activation gating variable
                self.m_init.value = self.initial_conditions['m'] # Na+ activation gating variable
                self.h_init.value = self.initial_conditions['h'] # Na+ inactivation gating variable
            else:
                self.phi_m_n_init.value = self.initial_conditions['phi_m_n'] # Neuronal membrane potential
                self.phi_m_g_init.value = self.initial_conditions['phi_m_g'] # Glial membrane potential
                self.Na_i_n_init.value = self.initial_conditions['Na_i_n'] # Neuronal intracellular Na+ concentration
                self.Na_i_g_init.value = self.initial_conditions['Na_i_g'] # Glial intracellular Na+ concentration
                self.Na_e_init.value = self.initial_conditions['Na_e']   # Extracellular Na+ concentration
                self.K_i_n_init.value = self.initial_conditions['K_i_n'] # Neuronal intracellular K+ concentration
                self.K_i_g_init.value = self.initial_conditions['K_i_g'] # Glial intracellular K+ concentration
                self.K_e_init.value = self.initial_conditions['K_e']  # Extracellular K+ concentration
                self.Cl_i_n_init.value = self.initial_conditions['Cl_i_n'] # Neuronal intracellular Cl- concentration
                self.Cl_i_g_init.value = self.initial_conditions['Cl_i_g'] # Glial intracellular Cl- concentration
                self.Cl_e_init.value = self.initial_conditions['Cl_e'] # Extracellular Cl- concentration
                self.n_init.value = self.initial_conditions['n'] # K+ activation gating variable
                self.m_init.value = self.initial_conditions['m'] # Na+ activation gating variable
                self.h_init.value = self.initial_conditions['h'] # Na+ inactivation gating variable

        # Create function for the membrane potential at previous timestep
        self.phi_m_prev = dfx.fem.Function(self.V)
        self.phi_m_prev.name = "phi_m"

        # Solutions at previous timestep
        ui_p = self.u_p[0]
        ue_p = self.u_p[1]

        if self.MMS_test:
            self.phi_m_prev.interpolate(
                                dfx.fem.Expression(
                                    self.phi_m_init,
                                    self.V.element.interpolation_points()
                                )
                            )
            
            # Set intra- and extracellular potentials just for visualization   
            ui_space = ui_p[self.N_ions].function_space
            ue_space = ue_p[self.N_ions].function_space
            ui_p[self.N_ions].interpolate(
                                dfx.fem.Expression(
                                    self.phi_i_init,
                                    ui_space.element.interpolation_points()
                                )
                            )
            ue_p[self.N_ions].interpolate(
                                dfx.fem.Expression(
                                    self.phi_e_init,
                                    ue_space.element.interpolation_points()
                                )
                            )
        else:
            if self.glia_tags is None:
                # Only neuronal cells
                self.phi_m_prev.x.array[:] = self.phi_m_init.value
                print(f"Initial membrane potential: {self.phi_m_init.value}")
            else:
                # Both neuronal and glial cells
                self.neuron_dofs = dfx.fem.locate_dofs_topological(self.V, self.subdomains.dim, self.neuron_cells)
                self.glia_dofs = dfx.fem.locate_dofs_topological(self.V, self.subdomains.dim, self.glia_cells)

                self.phi_m_prev.x.array[self.neuron_dofs] = self.phi_m_n_init.value
                self.phi_m_prev.x.array[self.glia_dofs] = self.phi_m_g_init.value

                print(f"Initial neuronal membrane potential: {self.phi_m_n_init.value}")
                print(f"Initial glial membrane potential: {self.phi_m_g_init.value}")

        # Set initial concentrations
        for idx, ion in enumerate(self.ion_list):
            # Set initial value of intra- and extracellular ion concentrations
            if self.MMS_test:
                ui_space = ui_p[idx].function_space
                ue_space = ue_p[idx].function_space
                ui_p[idx].interpolate(
                            dfx.fem.Expression(
                                ion['ki_init'],
                                ui_space.element.interpolation_points()
                            )
                        )
                ue_p[idx].interpolate(
                            dfx.fem.Expression(
                                ion['ke_init'],
                                ue_space.element.interpolation_points()
                            )
                        )
            else:
                if self.glia_tags is None:
                    # Set the array values at the subspace dofs 
                    ui_p[idx].x.array[:] = ion['ki_init'].value
                    ue_p[idx].x.array[:] = ion['ke_init'].value

                    print(f"Initial condition for {ion['name']}_i set to {ion['ki_init'].value}")
                    print(f"Initial condition for {ion['name']}_e set to {ion['ke_init'].value}")
                else:
                    # Set the array values at the subspace dofs 
                    ui_p[idx].x.array[self.neuron_dofs] = ion['ki_init_n'].value
                    ui_p[idx].x.array[self.glia_dofs]   = ion['ki_init_g'].value
                    ue_p[idx].x.array[:] = ion['ke_init'].value

                    print(f"Initial condition for {ion['name']}_i_n: {ion['ki_init_n'].value}")
                    print(f"Initial condition for {ion['name']}_i_g: {ion['ki_init_g'].value}")
                    print(f"Initial condition for {ion['name']}_e: {ion['ke_init'].value}")
        
        print("Initial conditions set.")
            
    def setup_variational_form(self):

        print("Setting up variational form ...")
        
        # Check that ionic models have been initialized
        if len(self.ionic_models)==0 and self.comm.rank==0:
            raise RuntimeError('\nNo ionic model(s) specified.\nCall init_ionic_model() to provide ionic models.\n')

        # Aliases
        dt  = self.dt # Timestep
        F   = self.F  # Faraday constant
        C_M = self.C_M # Membrane capacitance
        psi = self.psi # RT / F

        # Restrictions for traces
        i_res = "+"  # Intracellular facet restriction
        e_res = "-"  # Extracellular facet restriction

        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)
        dS  = self.dS(self.gamma_tags)
            
        # Trial and test functions
        u, v = ufl.TrialFunctions(self.W), ufl.TestFunctions(self.W) # Trial and test functions of the mixed space
        ui, vi = u[:self.num_variables], v[:self.num_variables] # Intracellular trial and test functions
        ue, ve = u[self.num_variables:], v[self.num_variables:] # Extracellular trial and test functions

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
        alpha_i_sum = ufl.ZeroBaseForm(None) # Sum of fractions intracellular
        alpha_e_sum = ufl.ZeroBaseForm(None) # Sum of fractions extracellular
        J_phi_i     = ufl.ZeroBaseForm(None) # Total intracellular flux
        J_phi_e     = ufl.ZeroBaseForm(None) # Total extracellular flux

        # Total ionic channel current
        I_ch = dict.fromkeys(self.gamma_tags, ufl.ZeroBaseForm(None))

        # Initialize parts of variational formulation
        for idx, ion in enumerate(self.ion_list):

            # Get ion attributes
            z  = ion['z']
            Di = ion['Di']
            De = ion['De']

            # Add ion specific contribution to fraction alpha
            alpha_i_sum += Di * z**2 * ui_p[idx]
            alpha_e_sum += De * z**2 * ue_p[idx]
            
            # Calculate and update Nernst potential for current ion
            ion['E'] = (psi/z) * ufl.ln(ue_p[idx] / ui_p[idx])

            # Initialize dictionary of ionic channel
            ion['I_ch'] = dict.fromkeys(self.gamma_tags, ufl.ZeroBaseForm(None))

            # Loop over ionic models
            for model in self.ionic_models:

                # Loop over ionic model tags
                for gamma_tag in model.tags:

                    I_ch_k_ = model._eval(idx)
                    
                    # Add stimulus current if the current cell membrane belongs to
                    # a cell that is stimulated
                    if (gamma_tag in self.stimulus_tags
                        and ion['name']=='Na'
                        and isinstance(model, HodgkinHuxley)
                        ):
                        
                        if self.stimulus_region:
                            stim: ufl.Coefficient = model._add_stimulus(idx, step=True, range=self.stimulus_region_range, dir=self.stimulus_region_direction)
                        else:
                            stim: ufl.Coefficient = model._add_stimulus(idx, step=True)
                        
                        I_ch_k_ += stim

                        # Store stimulus expression for output
                        self.stim_ufl_expr = stim

                    # Add contributions to ionic channel current
                    ion['I_ch'][gamma_tag] += I_ch_k_

                    # Add contribution to total channel current
                    I_ch[gamma_tag] += I_ch_k_

        # Initialize variational form
        a = ufl.ZeroBaseForm(None)
        L = ufl.ZeroBaseForm(None)

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
            ki_prev = ui_p[idx] # Previous solution

            # Set extracellular ion attributes
            ke  = ue[idx]       # Trial function
            vke = ve[idx]       # Test function
            ke_prev = ue_p[idx] # Previous solution

            # Set fraction of ion-specific intra- and extracellular I_cap
            alpha_i = Di * z**2 * ki_prev / alpha_i_sum
            alpha_e = De * z**2 * ke_prev / alpha_e_sum

            # Linearized ion fluxes
            Ji = -Di * grad(ki) - (Di*z/psi) * ki_prev * grad(phi_i)
            Je = -De * grad(ke) - (De*z/psi) * ke_prev * grad(phi_e)
            
            # Add contributions to total current flux
            J_phi_i += z*Ji
            J_phi_e += z*Je

            # A couple of useful constants
            C_i = C_M * alpha_i(i_res) / (F*z)
            C_e = C_M * alpha_e(e_res) / (F*z)

            # Weak form - equation for k_i
            a += ki*vki*dxi - dt * inner(Ji, grad(vki)) * dxi
            a += C_i * inner(phi_i(i_res), vki(i_res)) * dS
            a -= C_i * inner(phi_e(e_res), vki(i_res)) * dS
            L += ki_prev*vki*dxi
            
            # Weak form - equation for k_e
            a += ke*vke*dxe - dt * inner(Je, grad(vke)) * dxe
            a += C_e * inner(phi_e(e_res), vke(e_res)) * dS 
            a -= C_e * inner(phi_i(i_res), vke(e_res)) * dS 
            L += ke_prev*vke*dxe

            # Ionic channels
            for gamma_tag in self.gamma_tags:
                L -= 1/(F*z) * (dt*I_ch_k[gamma_tag] - alpha_i(i_res)*C_M*self.phi_m_prev) * vki(i_res) * dS(gamma_tag)
                L += 1/(F*z) * (dt*I_ch_k[gamma_tag] - alpha_e(e_res)*C_M*self.phi_m_prev) * vke(e_res) * dS(gamma_tag)

            # Source terms
            L += dt * inner(ion['f_i'], vki) * dxi
            L += dt * inner(ion['f_e'], vke) * dxe

            if self.MMS_test:
                # Concentrations source terms
                L += dt * inner(ion['f_k_i'], vki) * dxi # Equation for k_i
                L += dt * inner(ion['f_k_e'], vke) * dxe # Equation for k_e
                
                # Enforcing correction for I_m
                L += dt/(F*z) * alpha_i(i_res) * inner(ion['f_I_m'], vki(i_res)) * dS(self.gamma_tags)
                L -= dt/(F*z) * alpha_e(e_res) * inner(ion['f_I_m'], vke(e_res)) * dS(self.gamma_tags)
            
                # Enforcing correction for I_m, assuming gM_k = gM / N_ions
                L -= dt/(F*z) * alpha_e(e_res)*inner(self.src_terms['f_gamma'], vke(e_res))*dS(self.gamma_tags)

                # Exterior boundary terms (zero in "physical problem")
                L -=  dt * dot(ion['J_k_e'], self.n_outer) * vke * self.ds # Equation for k_e
                L += F*z * dot(ion['J_k_e'], self.n_outer) * vphi_e * self.ds # Equation for phi_e

        # Weak form - potential equations
        a -= dt * inner(J_phi_i, grad(vphi_i)) * dxi
        a -= dt * inner(J_phi_e, grad(vphi_e)) * dxe

        # Weak form - trace terms
        a += C_M/F * (phi_i(i_res) - phi_e(e_res)) * vphi_i(i_res) * dS
        a += C_M/F * (phi_e(e_res) - phi_i(i_res)) * vphi_e(e_res) * dS

        for gamma_tag in self.gamma_tags:
            L += 1/F * (dt*I_ch[gamma_tag] - C_M*self.phi_m_prev) * (vphi_e(e_res) - vphi_i(i_res)) * dS(gamma_tag)

        if self.MMS_test:
            # Phi source terms
            L -= dt * inner(self.src_terms['f_phi_i'], vphi_i) * dxi # Equation for phi_i
            L -= dt * inner(self.src_terms['f_phi_e'], vphi_e) * dxe # Equation for phi_e

            # Enforcing correction for I_m
            L += dt * inner(self.src_terms['f_phi_m'], vphi_i(i_res) - vphi_e(e_res)) * dS(self.gamma_tags)
            L -= dt * inner(self.src_terms['f_gamma'], vphi_e(e_res)) * dS(self.gamma_tags)

        # Extract form block and compile C++ forms
        self.a = dfx.fem.form(ufl.extract_blocks(a), jit_options=self.jit_parameters)
        self.L = dfx.fem.form(ufl.extract_blocks(L), jit_options=self.jit_parameters)

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
        u, v = ufl.TrialFunctions(self.W), ufl.TestFunctions(self.W) # Trial and test functions of the mixed space
        ui, vi = u[:self.num_variables], v[:self.num_variables] # Intracellular trial and test functions
        ue, ve = u[self.num_variables:], v[self.num_variables:] # Extracellular trial and test functions

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
        P = ufl.ZeroBaseForm(None)

        # Setup diagonal preconditioner
        for idx, ion in enumerate(self.ion_list):

            # Get ion attributes
            z    = ion['z']
            Di   = ion['Di']
            De   = ion['De']

            # Set intracellular ion attributes
            ki  = ui[idx]       # Trial function
            vki = vi[idx]       # Test function
            ki_prev = ui_p[idx] # Previous solution

            # Set extracellular ion attributes
            ke  = ue[idx]       # Trial function
            vke = ve[idx]       # Test function
            ke_prev = ue_p[idx] # Previous solution

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
            P += ki*vki*dxi + dt * inner(Di*grad(ki), grad(vki)) * dxi

            # weak form - equation for k_e
            P += ke*vke*dxe + dt * inner(De*grad(ke), grad(vke)) * dxe

        # Add flux contributions to weak form equations
        P -= dt * inner(J_phi_i, grad(vphi_i)) * dxi + (C_M/F) * inner(phi_i('+'), vphi_i('+')) * dS
        P -= dt * inner(J_phi_e, grad(vphi_e)) * dxe + (C_M/F) * inner(phi_e('-'), vphi_e('-')) * dS        

        # Create block preconditioner matrix
        P = ufl.extract_blocks(P)

        # Convert to C++ form
        self.P = dfx.fem.form(P, jit_options=self.jit_parameters)

    def setup_MMS_params(self):

        print('Setting up MMS parameters ...')

        assert np.allclose([self.C_M.value, self.R.value, self.F.value, self.psi.value], [1.0]*4) 

        self.M = ExactSolutionsKNPEMI(self.mesh, self.t)
        self.exact_sols, self.src_terms = self.M.get_mms_terms()

        # initial values
        self.phi_i_init = self.exact_sols["phi_i_init"] # internal potential (V) just for visualization
        self.phi_e_init = self.exact_sols["phi_e_init"] # external potential (V) just for visualization
        self.phi_m_init = self.phi_i_init - self.phi_e_init # membrane potential (V)

        # create ions
        self.Na = {'Di' : dfx.fem.Constant(self.mesh, 1.0),
                   'De' : dfx.fem.Constant(self.mesh, 1.0),
                   'z'  : dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1)),
                   'ki_init':self.exact_sols['Na_i'],
                   'ke_init':self.exact_sols['Na_e'],		
                   'f_k_i':self.src_terms['f_Na_i'],
                   'f_k_e':self.src_terms['f_Na_e'],
                   'J_k_e':self.src_terms['J_Na_e'],
                   'f_I_m':self.src_terms['f_phi_Na'],
                   'name':'Na',
                   'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                   'f_e' : dfx.fem.Constant(self.mesh, 0.0)
                }

        self.K = {'Di' : dfx.fem.Constant(self.mesh, 1.0),
                  'De' : dfx.fem.Constant(self.mesh, 1.0),
                  'z'  : dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1)),
                  'ki_init':self.exact_sols['K_i'],
                  'ke_init':self.exact_sols['K_e'],			 
                  'f_k_i':self.src_terms['f_K_i'],
                  'f_k_e':self.src_terms['f_K_e'],
                  'J_k_e':self.src_terms['J_K_e'],
                  'f_I_m':self.src_terms['f_phi_K'],
                  'name':'K',
                  'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                  'f_e' : dfx.fem.Constant(self.mesh, 0.0)
                }

        self.Cl = {'Di' : dfx.fem.Constant(self.mesh,  1.0),
                   'De' : dfx.fem.Constant(self.mesh,  1.0),
                   'z'  : dfx.fem.Constant(self.mesh, dfx.default_scalar_type(-1)),
                   'ki_init':self.exact_sols['Cl_i'],
                   'ke_init':self.exact_sols['Cl_e'],		
                   'f_k_i':self.src_terms['f_Cl_i'],
                   'f_k_e':self.src_terms['f_Cl_e'],
                   'J_k_e':self.src_terms['J_Cl_e'],
                   'f_I_m':self.src_terms['f_phi_Cl'],
                   'name':'Cl',
                   'f_i' : dfx.fem.Constant(self.mesh, 0.0),
                   'f_e' : dfx.fem.Constant(self.mesh, 0.0)
                }

        # create ion list
        self.ion_list = [self.Na, self.K, self.Cl]
        self.N_ions   = len(self.ion_list) 

    def print_conservation(self):

        # Define measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        Na_i = self.wh[0][0]
        K_i  = self.wh[0][1]
        Cl_i = self.wh[0][2]
        
        Na_e = self.wh[1][0]
        K_e  = self.wh[1][1]
        Cl_e = self.wh[1][2]

        Na_tot = dfx.fem.assemble_scalar(dfx.fem.form(Na_i*dxi + Na_e*dxe, jit_options=self.jit_parameters))
        K_tot  = dfx.fem.assemble_scalar(dfx.fem.form(K_i *dxi + K_e *dxe, jit_options=self.jit_parameters))
        Cl_tot = dfx.fem.assemble_scalar(dfx.fem.form(Cl_i*dxi + Cl_e*dxe, jit_options=self.jit_parameters))

        print(f"Time {self.t.value*1e3:.2f} ms")
        print(f"Total Na+ concentration: {self.comm.allreduce(Na_tot, op=MPI.SUM):.2e} mol")
        print(f"Total K+  concentration: {self.comm.allreduce(K_tot,  op=MPI.SUM):.2e} mol")
        print(f"Total Cl- concentration: {self.comm.allreduce(Cl_tot, op=MPI.SUM):.2e} mol")

        for tag in self.intra_tags:
            vol_i = dfx.fem.assemble_scalar(dfx.fem.form(1*dxi(tag), jit_options=self.jit_parameters))
            vol_i = self.comm.allreduce(vol_i, op=MPI.SUM)
            area_i = dfx.fem.assemble_scalar(dfx.fem.form(1*self.dS(tag), jit_options=self.jit_parameters))
            area_i = self.comm.allreduce(area_i, op=MPI.SUM)
            N_Na_i = dfx.fem.assemble_scalar(dfx.fem.form(Na_i*dxi(tag), jit_options=self.jit_parameters))
            N_Na_i = self.comm.allreduce(N_Na_i, op=MPI.SUM)
            N_K_i = dfx.fem.assemble_scalar(dfx.fem.form(K_i*dxi(tag), jit_options=self.jit_parameters))
            N_K_i = self.comm.allreduce(N_K_i, op=MPI.SUM)
            N_Cl_i = dfx.fem.assemble_scalar(dfx.fem.form(Cl_i*dxi(tag), jit_options=self.jit_parameters))
            N_Cl_i = self.comm.allreduce(N_Cl_i, op=MPI.SUM)

            charge = (N_Na_i + N_K_i - N_Cl_i) * self.F.value
            print(f"  Intra tag {tag}: Volume = {vol_i:.2e} m^3, Area = {area_i:.2e} m^2, Charge = {charge:.2e} C")

    def print_errors(self):
        
        exact_sols = self.exact_sols
        
        # Define integral measures
        dxi = self.dx(self.intra_tags)
        dxe = self.dx(self.extra_tag)

        Na_i  = self.wh[0][0]
        K_i   = self.wh[0][1]
        Cl_i  = self.wh[0][2]
        phi_i = self.wh[0][3]

        Na_e  = self.wh[1][0]
        K_e   = self.wh[1][1]
        Cl_e  = self.wh[1][2]
        phi_e = self.wh[1][3]
        
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
        self.C_M = Constant(self.mesh, dfx.default_scalar_type(self.C_M_value))  # Capacitance [F / m^2]
        self.T   = Constant(self.mesh, dfx.default_scalar_type(self.T_value))   # Temperature [K]
        self.F   = Constant(self.mesh, dfx.default_scalar_type(self.F_value)) # Faraday's constant [C/mol]
        self.R   = Constant(self.mesh, dfx.default_scalar_type(self.R_value)) # Gas constant [J/(K*mol)]
        self.psi = Constant(self.mesh, dfx.default_scalar_type(self.psi_value)) # Recurring variable
    
        self.g_Na_bar  = Constant(self.mesh, dfx.default_scalar_type(self.g_Na_bar_val)) # Na max conductivity [S/m**2]
        self.g_K_bar   = Constant(self.mesh, dfx.default_scalar_type(self.g_K_bar_val))  # K max conductivity [S/m**2]
        self.g_Na_leak   = Constant(self.mesh, dfx.default_scalar_type(self.g_Na_leak_val)) # Na leak conductivity [S/m**2]
        self.g_Na_leak_g = Constant(self.mesh, dfx.default_scalar_type(self.g_Na_leak_g_val)) # Na leak conductivity [S/m**2]
        self.g_K_leak    = Constant(self.mesh, dfx.default_scalar_type(self.g_K_leak_val)) # K leak conductivity [S/m**2]
        self.g_K_leak_g  = Constant(self.mesh, dfx.default_scalar_type(self.g_K_leak_g_val)) # K leak conductivity [S/m**2]
        self.g_Cl_leak   = Constant(self.mesh, dfx.default_scalar_type(self.g_Cl_leak_val)) # Cl leak conductivity [S/m**2]
        self.g_Cl_leak_g = Constant(self.mesh, dfx.default_scalar_type(self.g_Cl_leak_g_val)) # Cl leak conductivity [S/m**2]
        self.g_syn_bar = Constant(self.mesh, dfx.default_scalar_type(self.g_syn_bar_val)) # Synaptic conductivity [S/m**2]
        self.a_syn     = Constant(self.mesh, dfx.default_scalar_type(self.a_syn_val)) # Synaptic time constant [s]
        self.T_stim    = Constant(self.mesh, dfx.default_scalar_type(self.T_stim_val)) # Synaptic stimulus time [s]
        self.D_Na = Constant(self.mesh, dfx.default_scalar_type(1.33e-9)) # Diffusion coefficients Na [m/s^2]
        self.D_K  = Constant(self.mesh, dfx.default_scalar_type(1.96e-9)) # Diffusion coefficients K [m/s^2]
        self.D_Cl = Constant(self.mesh, dfx.default_scalar_type(2.03e-9)) # Diffusion coefficients Cl [m/s^2]
        self.phi_rest  = Constant(self.mesh, dfx.default_scalar_type(-0.065)) # Resting membrane potential [V]

        # Potassium buffering params
        self.rho_pump = Constant(self.mesh, dfx.default_scalar_type(1.115e-6)) # maximum pump rate [mol/m**2 s]
        self.P_Nai = Constant(self.mesh, dfx.default_scalar_type(10))          # [Na+]i threshold for Na+/K+ pump [mol/m^3]
        self.P_Ke  = Constant(self.mesh, dfx.default_scalar_type(1.5))         # [K+]e  threshold for Na+/K+ pump [mol/m^3]
        self.k_dec = Constant(self.mesh, dfx.default_scalar_type(2.9e-8))	   # Decay factor for [K+]e [m/s]

        # Initial conditions
        self.phi_m_init = Constant(self.mesh, dfx.default_scalar_type(-0.070)) # Membrane potential [V] 
        self.Na_i_init  = Constant(self.mesh, dfx.default_scalar_type(12))     # Intracellular Na concentration [mol/m^3]
        self.Na_e_init  = Constant(self.mesh, dfx.default_scalar_type(140))    # Extracellular Na concentration [mol/m^3]
        self.K_i_init   = Constant(self.mesh, dfx.default_scalar_type(130))    # Intracellular K  concentration [mol/m^3]
        self.K_e_init   = Constant(self.mesh, dfx.default_scalar_type(4))      # Extracellular K  concentration [mol/m^3]
        self.Cl_i_init  = Constant(self.mesh, dfx.default_scalar_type(5))      # Intracellular Cl concentration [mol/m^3]
        self.Cl_e_init  = Constant(self.mesh, dfx.default_scalar_type(125))    # Extracellular Cl concentration [mol/m^3]

        # Neuro+glia
        self.phi_m_n_init = Constant(self.mesh, self.phi_m_init.value) # Membrane potential, neuronal [V] 
        self.phi_m_g_init = Constant(self.mesh, -0.085) # Membrane potential, glial [V]
        self.Na_i_n_init = Constant(self.mesh, self.Na_i_init.value) # Intracellular Na concentration, neuronal [mol/m^3]
        self.K_i_n_init  = Constant(self.mesh, self.K_i_init.value)  # Intracellular K  concentration, neuronal [mol/m^3]
        self.Cl_i_n_init = Constant(self.mesh, self.Cl_i_init.value) # Intracellular Cl concentration, neuronal [mol/m^3]
        self.Na_i_g_init = Constant(self.mesh, dfx.default_scalar_type(15))  # Intracellular Na concentration, glial [mol/m^3]
        self.K_i_g_init  = Constant(self.mesh, dfx.default_scalar_type(100)) # Intracellular K  concentration, glial [mol/m^3]
        self.Cl_i_g_init = Constant(self.mesh, dfx.default_scalar_type(5))   # Intracellular Cl concentration, glial [mol/m^3]

        # Initial values of gating variables
        self.n_init = Constant(self.mesh, dfx.default_scalar_type(0.276))
        self.m_init = Constant(self.mesh, dfx.default_scalar_type(0.0379))
        self.h_init = Constant(self.mesh, dfx.default_scalar_type(0.688))

        # Source terms
        self.Na_e_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Na_i_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.K_e_f  = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.K_i_f  = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Cl_e_f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.Cl_i_f = Constant(self.mesh, dfx.default_scalar_type(0.0))

        # Ion dictionaries and list
        self.Na = {'g_leak':self.g_Na_leak, 'g_leak_g':self.g_Na_leak_g, 'Di':self.D_Na, 'De':self.D_Na, 'ki_init':self.Na_i_init, 'ke_init':self.Na_e_init, 'ki_init_n' : self.Na_i_n_init, 'ki_init_g' : self.Na_i_g_init, 'z':Constant(self.mesh, 1.0),  'f_e': self.Na_e_f, 'f_i':self.Na_i_f, 'name':'Na', 'rho_p': Constant(self.mesh, dfx.default_scalar_type(3*self.rho_pump.value))}
        self.K  = {'g_leak':self.g_K_leak,  'g_leak_g':self.g_K_leak_g, 'Di':self.D_K,  'De':self.D_K,  'ki_init':self.K_i_init,  'ke_init':self.K_e_init,  'ki_init_n' : self.K_i_n_init, 'ki_init_g' : self.K_i_g_init,  'z':Constant(self.mesh, 1.0),  'f_e': self.K_e_f,  'f_i':self.K_i_f,  'name':'K' , 'rho_p': Constant(self.mesh, dfx.default_scalar_type(-2*self.rho_pump.value))}
        self.Cl = {'g_leak':self.g_Cl_leak, 'g_leak_g':self.g_Cl_leak_g, 'Di':self.D_Cl, 'De':self.D_Cl, 'ki_init':self.Cl_i_init, 'ke_init':self.Cl_e_init,  'ki_init_n' : self.Cl_i_n_init, 'ki_init_g' : self.Cl_i_g_init, 'z':Constant(self.mesh, -1.0), 'f_e': self.Cl_e_f, 'f_i':self.Cl_i_f, 'name':'Cl', 'rho_p': Constant(self.mesh, 0.0)}
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
