import ufl
import numpy.typing

import numpy   as np
import dolfinx as dfx
import sympy   as sp

from CGx.utils.misc           import mark_boundaries_cube_MMS, mark_boundaries_square_MMS
from sympy.utilities.lambdify import lambdify

class SetupMMS:
    """ Class for calculating source terms of the EMI or the KNP-EMI system for given exact
    solutions. """
    def __init__(self, mesh: dfx.mesh.Mesh):
        self.mesh = mesh
        self.dim  = mesh.topology.dim
        # define symbolic variables
        if self.dim==2:
            self.x, self.y, self.t = sp.symbols('x[0] x[1] t')
        elif self.dim==3:
            self.x, self.y, self.z, self.t = sp.symbols('x[0] x[1] x[2] t')
    
    def get_MMS_terms_EMI_2D(self, time):
        """ Get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            EMI problem in two dimensions. """

        # Forcing factor expressions
        class IntracellularSource:
            def __init__(self, t_0: float):
                self.t = t_0 # Initial time

            def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
                return 8*np.pi**2 * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * (1.0 + np.exp(-self.t))

        class ExtracellularSource:
            def __init__(self): pass

            def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
                return 8*np.pi**2 * np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

        # Exact solution expressions
        class uiExact:
            def __init__(self, t_0: float):
                self.t = t_0

            def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
                return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * (1.0 + np.exp(-self.t))

        class ueExact:
            def __init__(self): pass

            def __call__(self, x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.float64]:
                return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

        # Create P1 space for all functions
        V = dfx.fem.functionspace(self.mesh, ("Lagrange", 1))

        # Create functions for storing exact solutions of the four quantities
        # intracellular potential, extracellular potential, intracellular source term
        # and extracellular source term.
        exact_functions = [dfx.fem.Function(V) for _ in range(4)] # ui, ue, fi, fe
        ui_expr = uiExact(t_0=time)
        ue_expr = ueExact()
        fi_expr = IntracellularSource(t_0=time)
        fe_expr = ExtracellularSource()
        expressions = [ui_expr, ue_expr, fi_expr, fe_expr]
        [exact_functions[i].interpolate(expressions[i]) for i in range(4)] # Interpolate function expressions
        
        # Gather expressions
        # exact solutions
        exact_sols = {'phi_i' : exact_functions[0],
                      'phi_e' : exact_functions[1]}
        # source terms
        src_terms = {'f_i' : exact_functions[2],
                     'f_e' : exact_functions[3]}

        return exact_sols, src_terms

    def get_exact_solution_KNPEMI(self):
        """ Define manufactured (exact) solutions for the KNP-EMI system. The functions
            considered are sodium, potassium and chloride ion concentrations, as well as
            electric potentials in the intra- and extracellular domain.
        """
        x = self.x; y = self.y; t = self.t

        if self.dim==2:
            # sodium (Na) concentration
            Na_i_e = 0.7 + 0.3*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            Na_e_e = 1.0 + 0.6*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            # potassium (K) concentration
            K_i_e = 0.3 + 0.3*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            K_e_e = 1.0 + 0.2*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            # chloride (Cl) concentration
            Cl_i_e = 1.0 + 0.6*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            Cl_e_e = 2.0 + 0.8*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.exp(-t)
            # potentials
            phi_i_e = sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)*(1 + sp.exp(-t))
            phi_e_e = sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)

        elif self.dim==3:
            z = self.z
            # sodium (Na) concentration
            Na_i_e = 0.7 + 0.3*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            Na_e_e = 1.0 + 0.6*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            # potassium (K) concentration
            K_i_e = 0.3 + 0.3*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            K_e_e = 1.0 + 0.2*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            # chloride (Cl) concentration
            Cl_i_e = 1.0 + 0.6*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            Cl_e_e = 2.0 + 0.8*sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z)*sp.exp(-t)
            # potentials
            phi_i_e = sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)*sp.cos(2*np.pi*z)*(1 + sp.exp(-t))
            phi_e_e = sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)*sp.cos(2*np.pi*z)

        exact_solutions = {'Na_i_e':Na_i_e, 'K_i_e':K_i_e, 'Cl_i_e':Cl_i_e,
                           'Na_e_e':Na_e_e, 'K_e_e':K_e_e, 'Cl_e_e':Cl_e_e,
                           'phi_i_e':phi_i_e, 'phi_e_e':phi_e_e}

        return exact_solutions

    def get_MMS_terms_KNPEMI_2D(self, time):
        """ Get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            KNP-EMI problem in two dimensions. """
        # Variables
        x = self.x; y = self.y; t = self.t

        # get manufactured solution
        exact_solutions = self.get_exact_solution_KNPEMI()
        # unwrap exact solutions
        for key in exact_solutions:
            # exec() changed from python2 to python3
            exec('global %s; %s = exact_solutions["%s"]' % (key, key ,key))

        # Calculate components
        # gradients
        grad_Nai, grad_Ki, grad_Cli, grad_phii, grad_Nae, grad_Ke, grad_Cle, grad_phie = \
                [np.array([sp.diff(foo, x),  sp.diff(foo, y)])
                for foo in (Na_i_e, K_i_e, Cl_i_e, phi_i_e, Na_e_e, K_e_e, Cl_e_e, phi_e_e)]

        # compartmental fluxes
        J_Na_i = - grad_Nai - Na_i_e*grad_phii
        J_Na_e = - grad_Nae - Na_e_e*grad_phie
        J_K_i  = - grad_Ki  - K_i_e*grad_phii
        J_K_e  = - grad_Ke  - K_e_e*grad_phie
        J_Cl_i = - grad_Cli + Cl_i_e*grad_phii
        J_Cl_e = - grad_Cle + Cl_e_e*grad_phie

        # membrane potential
        phi_m_e = phi_i_e - phi_e_e

        # total membrane flux defined intracellularly (F sum_k z^k J^k_i)
        total_flux_i = J_Na_i + J_K_i - J_Cl_i
        # current defined intracellular: total_flux_i dot i_normals
        # [(-1, 0), (1, 0), (0, -1), (0, 1)]
        JMe_i = [- total_flux_i[0], total_flux_i[0], - total_flux_i[1], total_flux_i[1]]

        # membrane flux defined extracellularly ( - F sum_k z^k J^k_i)
        total_flux_e = - (J_Na_e + J_K_e - J_Cl_e)
        # current defined intracellular: total_flux_e dot e_normals
        # [(1, 0), (-1, 0), (0, 1), (0, -1)]
        JMe_e = [total_flux_e[0], - total_flux_e[0], total_flux_e[1], - total_flux_e[1]]
        
        # ion channel currents
        I_ch_Na = phi_m_e                 # Na
        I_ch_K  = phi_m_e                 # K
        I_ch_Cl = phi_m_e                 # Cl
        I_ch = I_ch_Na + I_ch_K + I_ch_Cl # total 

        # Calculate source terms
        # equations for ion cons: f = dk_r/dt + div (J_kr)
        f_Na_i = sp.diff(Na_i_e, t) + sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y)
        f_Na_e = sp.diff(Na_e_e, t) + sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y)
        f_K_i  = sp.diff(K_i_e, t)  + sp.diff(J_K_i[0], x)  + sp.diff(J_K_i[1], y)
        f_K_e  = sp.diff(K_e_e, t)  + sp.diff(J_K_e[0], x)  + sp.diff(J_K_e[1], y)
        f_Cl_i = sp.diff(Cl_i_e, t) + sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y)
        f_Cl_e = sp.diff(Cl_e_e, t) + sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y)

        # equations for potentials: fE = - F sum(z_k*div(J_k_r)
        f_phi_i = - ((sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y))
                  + ( sp.diff(J_K_i[0],  x) + sp.diff(J_K_i[1],  y))
                  - ( sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y)))
        f_phi_e = - ((sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y))
                  + ( sp.diff(J_K_e[0],  x) + sp.diff(J_K_e[1],  y))
                  - ( sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y)))

        # equation for phi_m: f = C_M*d(phi_m)/dt - (I_M - I_ch) where we have
        # chosen I_M = F sum_k z^k J_i^k n_i = total_flux_i
        fJM = [sp.diff(phi_m_e, t) + I_ch - foo for foo in JMe_i]
        # coupling condition for I_M: (total_flux_i*n_i) = (- total_flux_e*n_e) + f
        # giving f = total_flux_i*n_i + total_flux_e*n_e
        fgM = [i - e for i, e in zip(JMe_i, JMe_e)]

        ##### Convert to expressions with exact solutions #####
        # Create P1 space for all functions
        V = dfx.fem.functionspace(self.mesh, ("Lagrange", 1))

        # Ion concentrations and electric potentials
        var_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_i_e, phi_e_e, phi_m_e]
        var_exprs = [SymPyToDOLFINxExpr([x, y], time, var_func, self.dim) for var_func in var_sym_funcs]
        var_dfx_funcs = [dfx.fem.Function(V) for _ in var_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(var_dfx_funcs, var_exprs)]
        Nai_e, Nae_e, Ki_e, Ke_e, Cli_e, Cle_e, phii_e, phie_e, phiM_e = [var_dfx_funcs[i] for i in range(len(var_dfx_funcs))]

        # Membrane flux
        JM_e_exprs = [SymPyToDOLFINxExpr([x, y], time, JMe_func, self.dim) for JMe_func in JMe_i]
        JM_e = [dfx.fem.Function(V) for _ in range(len(JM_e_exprs))]
        [JM_e[i].interpolate(JM_e_exprs[i]) for i in range(len(JM_e_exprs))]

        # source terms
        source_sym_funcs = [f_Na_i, f_Na_e, f_K_i, f_K_e, f_Cl_i, f_Cl_e, f_phi_i, f_phi_e]
        source_exprs = [SymPyToDOLFINxExpr([x, y], time, source_func, self.dim) for source_func in source_sym_funcs]
        source_dfx_funcs = [dfx.fem.Function(V) for _ in source_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(source_dfx_funcs, source_exprs)]
        f_Nai, f_Nae, f_Ki, f_Ke, f_Cli, f_Cle, f_phii, f_phie = [source_dfx_funcs[i] for i in range(len(source_dfx_funcs))]

        # source term membrane flux
        f_JM_exprs = [SymPyToDOLFINxExpr([x, y], time, fJM_func, self.dim) for fJM_func in fJM]
        f_JM = [dfx.fem.Function(V) for _ in f_JM_exprs]
        [f_JM[i].interpolate(f_JM_exprs[i]) for i in range(len(f_JM_exprs))]

        # source term continuity coupling condition on gamma
        f_gM_exprs = [SymPyToDOLFINxExpr([x, y], time, fgM_func, self.dim) for fgM_func in fgM]
        f_gM = [dfx.fem.Function(V) for _ in f_gM_exprs]
        [f_gM[i].interpolate(f_gM_exprs[i]) for i in range(len(f_gM_exprs))]

        # initial conditions concentrations
        init_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_m_e]
        init_exprs = [SymPyToDOLFINxExpr([x, y], time, init_func, self.dim) for init_func in init_sym_funcs]
        init_dfx_funcs = [dfx.fem.Function(V) for _ in init_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(init_dfx_funcs, init_exprs)]
        init_Nai, init_Nae, init_Ki, init_Ke, init_Cli, init_Cle, init_phiM = [init_dfx_funcs[i] for i in range(len(init_dfx_funcs))]

        # exterior boundary terms
        P1_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), degree=1)
        V_vec = dfx.fem.functionspace(self.mesh, element=P1_vec)
        ext_sym_funcs = [J_Na_e, J_K_e, J_Cl_e]
        ext_exprs = [SymPyToDOLFINxExpr([x, y], time, ext_func, self.dim) for ext_func in ext_sym_funcs]
        ext_dfx_funcs = [dfx.fem.Function(V_vec) for _ in ext_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ext_dfx_funcs, ext_exprs)]
        J_Nae, J_Ke, J_Cle = [ext_dfx_funcs[i] for i in range(len(ext_dfx_funcs))]

        # ion channel currents
        ion_ch_sym_funcs = [I_ch_Na, I_ch_K, I_ch_Cl]
        ion_ch_exprs = [SymPyToDOLFINxExpr([x, y], time, ion_ch_func, self.dim) for ion_ch_func in ion_ch_sym_funcs]
        ion_ch_funcs = [dfx.fem.Function(V) for _ in ion_ch_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ion_ch_funcs, ion_ch_exprs)]
        I_ch_Na, I_ch_K, I_ch_Cl = [ion_ch_funcs[i] for i in range(len(ion_ch_funcs))]

        # Gather expressions
        # exact solutions
        exact_sols = {'Na_i_e':Nai_e, 'K_i_e':Ki_e, 'Cl_i_e':Cli_e,
                      'Na_e_e':Nae_e, 'K_e_e':Ke_e, 'Cl_e_e':Cle_e,
                      'phi_i_e':phii_e, 'phi_e_e':phie_e, 'phi_m_e':phiM_e,
                      'I_M_e':JM_e, 'I_ch_Na':I_ch_Na, 'I_ch_K':I_ch_K,
                      'I_ch_Cl':I_ch_Cl}
        # source terms
        src_terms = {'f_Na_i':f_Nai, 'f_K_i':f_Ki, 'f_Cl_i':f_Cli,
                     'f_Na_e':f_Nae, 'f_K_e':f_Ke, 'f_Cl_e':f_Cle,
                     'f_phi_i':f_phii, 'f_phi_e':f_phie, 'f_I_M':f_JM,
                     'f_g_M':f_gM}
        # initial conditions
        init_conds = {'Na_i':init_Nai, 'K_i':init_Ki, 'Cl_i':init_Cli,
                      'Na_e':init_Nae, 'K_e':init_Ke, 'Cl_e':init_Cle,
                      'phi_m':init_phiM}
        # boundary terms
        bndry_terms = {'J_Na_e':J_Nae, 'J_K_e':J_Ke, 'J_Cl_e':J_Cle}

        return src_terms, exact_sols, init_conds, bndry_terms

    def get_MMS_terms_KNPEMI_3D(self, time):
        """ Get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            KNP-EMI problem in three dimensions. """
        # Variables
        x = self.x; y = self.y; z = self.z; t = self.t

        # get manufactured solution
        exact_solutions = self.get_exact_solution()
        # unwrap exact solutions
        for key in exact_solutions:
            # exec() changed from python2 to python3
            exec('global %s; %s = exact_solutions["%s"]' % (key, key ,key))

        # Calculate components
        # gradients
        grad_Nai, grad_Ki, grad_Cli, grad_phii, grad_Nae, grad_Ke, grad_Cle, grad_phie = \
                [np.array([sp.diff(foo, x),  sp.diff(foo, y), sp.diff(foo, z)])
                for foo in (Na_i_e, K_i_e, Cl_i_e, phi_i_e, Na_e_e, K_e_e, Cl_e_e, phi_e_e)]

        # compartmental fluxes
        J_Na_i = - grad_Nai - Na_i_e*grad_phii
        J_Na_e = - grad_Nae - Na_e_e*grad_phie
        J_K_i  = - grad_Ki  - K_i_e*grad_phii
        J_K_e  = - grad_Ke  - K_e_e*grad_phie
        J_Cl_i = - grad_Cli + Cl_i_e*grad_phii
        J_Cl_e = - grad_Cle + Cl_e_e*grad_phie

        # membrane potential
        phi_m_e = phi_i_e - phi_e_e

        # total membrane flux defined intracellularly (F sum_k z^k J^k_i)
        total_flux_i = J_Na_i + J_K_i - J_Cl_i
        # current defined intracellular: total_flux_i dot i_normals
        # [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        JMe_i = [- total_flux_i[0], total_flux_i[0],
                 - total_flux_i[1], total_flux_i[1],
                 - total_flux_i[2], total_flux_i[2]]

        # membrane flux defined extracellularly ( - F sum_k z^k J^k_i)
        total_flux_e = - (J_Na_e + J_K_e - J_Cl_e)
        # current defined intracellular: total_flux_e dot e_normals
        # [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        JMe_e = [total_flux_e[0], - total_flux_e[0],
                 total_flux_e[1], - total_flux_e[1],
                 total_flux_e[2], - total_flux_e[2]]
        
        # ion channel currents
        I_ch_Na = phi_m_e                 # Na
        I_ch_K  = phi_m_e                 # K
        I_ch_Cl = phi_m_e                 # Cl
        I_ch = I_ch_Na + I_ch_K + I_ch_Cl # total 

        # Calculate source terms
        # equations for ion cons: f = dk_r/dt + div (J_kr)
        f_Na_i = sp.diff(Na_i_e, t) + sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y) + sp.diff(J_Na_i[2], z)
        f_Na_e = sp.diff(Na_e_e, t) + sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y) + sp.diff(J_Na_e[2], z)
        f_K_i  = sp.diff(K_i_e, t)  + sp.diff(J_K_i[0], x)  + sp.diff(J_K_i[1], y)  + sp.diff(J_K_i[2], z)
        f_K_e  = sp.diff(K_e_e, t)  + sp.diff(J_K_e[0], x)  + sp.diff(J_K_e[1], y)  + sp.diff(J_K_e[2], z)
        f_Cl_i = sp.diff(Cl_i_e, t) + sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y) + sp.diff(J_Cl_i[2], z)
        f_Cl_e = sp.diff(Cl_e_e, t) + sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y) + sp.diff(J_Cl_e[2], z)

        # equations for potentials: fE = - F sum(z_k*div(J_k_r)
        f_phi_i = - ((sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y) + sp.diff(J_Na_i[2], z))
                  + ( sp.diff(J_K_i[0],  x) + sp.diff(J_K_i[1],  y) + sp.diff(J_K_i[2],  z))
                  - ( sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y) + sp.diff(J_Cl_i[2], z)))
        f_phi_e = - ((sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y) + sp.diff(J_Na_e[2], z))
                  + ( sp.diff(J_K_e[0],  x) + sp.diff(J_K_e[1],  y) + sp.diff(J_K_e[2],  z))
                  - ( sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y) + sp.diff(J_Cl_e[2], z)))

        # equation for phi_m: f = C_M*d(phi_m)/dt - (I_M - I_ch) where we have
        # chosen I_M = F sum_k z^k J_i^k n_i = total_flux_i
        fJM = [sp.diff(phi_m_e, t) + I_ch - foo for foo in JMe_i]
        # coupling condition for I_M: (total_flux_i*n_i) = (- total_flux_e*n_e) + f
        # giving f = total_flux_i*n_i + total_flux_e*n_e
        fgM = [i - e for i, e in zip(JMe_i, JMe_e)]

        ##### Convert to expressions with exact solutions #####
        # Create P1 space for all functions
        V = dfx.fem.functionspace(self.mesh, ("Lagrange", 1))

        # Ion concentrations and electric potentials
        var_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_i_e, phi_e_e, phi_m_e]
        var_exprs = [SymPyToDOLFINxExpr([x, y], time, var_func, self.dim) for var_func in var_sym_funcs]
        var_dfx_funcs = [dfx.fem.Function(V) for _ in var_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(var_dfx_funcs, var_exprs)]
        Nai_e, Nae_e, Ki_e, Ke_e, Cli_e, Cle_e, phii_e, phie_e, phiM_e = [var_dfx_funcs[i] for i in range(len(var_dfx_funcs))]

        # Membrane flux
        JM_e_exprs = [SymPyToDOLFINxExpr([x, y], time, JMe_func, self.dim) for JMe_func in JMe_i]
        JM_e = [dfx.fem.Function(V) for _ in range(len(JM_e_exprs))]
        [JM_e[i].interpolate(JM_e_exprs[i]) for i in range(len(JM_e_exprs))]

        # source terms
        source_sym_funcs = [f_Na_i, f_Na_e, f_K_i, f_K_e, f_Cl_i, f_Cl_e, f_phi_i, f_phi_e]
        source_exprs = [SymPyToDOLFINxExpr([x, y], time, source_func, self.dim) for source_func in source_sym_funcs]
        source_dfx_funcs = [dfx.fem.Function(V) for _ in source_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(source_dfx_funcs, source_exprs)]
        f_Nai, f_Nae, f_Ki, f_Ke, f_Cli, f_Cle, f_phii, f_phie = [source_dfx_funcs[i] for i in range(len(source_dfx_funcs))]

        # source term membrane flux
        f_JM_exprs = [SymPyToDOLFINxExpr([x, y], time, fJM_func, self.dim) for fJM_func in fJM]
        f_JM = [dfx.fem.Function(V) for _ in f_JM_exprs]
        [f_JM[i].interpolate(f_JM_exprs[i]) for i in range(len(f_JM_exprs))]

        # source term continuity coupling condition on gamma
        f_gM_exprs = [SymPyToDOLFINxExpr([x, y], time, fgM_func, self.dim) for fgM_func in fgM]
        f_gM = [dfx.fem.Function(V) for _ in f_gM_exprs]
        [f_gM[i].interpolate(f_gM_exprs[i]) for i in range(len(f_gM_exprs))]

        # initial conditions concentrations
        init_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_m_e]
        init_exprs = [SymPyToDOLFINxExpr([x, y], time, init_func, self.dim) for init_func in init_sym_funcs]
        init_dfx_funcs = [dfx.fem.Function(V) for _ in init_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(init_dfx_funcs, init_exprs)]
        init_Nai, init_Nae, init_Ki, init_Ke, init_Cli, init_Cle, init_phiM = [init_dfx_funcs[i] for i in range(len(init_dfx_funcs))]

        # exterior boundary terms
        import ufl
        P1_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), degree=1)
        V_vec = dfx.fem.functionspace(self.mesh, element=P1_vec)
        ext_sym_funcs = [J_Na_e, J_K_e, J_Cl_e]
        ext_exprs = [SymPyToDOLFINxExpr([x, y], time, ext_func, self.dim) for ext_func in ext_sym_funcs]
        ext_dfx_funcs = [dfx.fem.Function(V_vec) for _ in ext_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ext_dfx_funcs, ext_exprs)]
        J_Nae, J_Ke, J_Cle = [ext_dfx_funcs[i] for i in range(len(ext_dfx_funcs))]

        # ion channel currents
        ion_ch_sym_funcs = [I_ch_Na, I_ch_K, I_ch_Cl]
        ion_ch_exprs = [SymPyToDOLFINxExpr([x, y], time, ion_ch_func, self.dim) for ion_ch_func in ion_ch_sym_funcs]
        ion_ch_funcs = [dfx.fem.Function(V) for _ in ion_ch_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ion_ch_funcs, ion_ch_exprs)]
        I_ch_Na, I_ch_K, I_ch_Cl = [ion_ch_funcs[i] for i in range(len(ion_ch_funcs))]

        # Gather expressions
        # exact solutions
        exact_sols = {'Na_i_e':Nai_e, 'K_i_e':Ki_e, 'Cl_i_e':Cli_e,
                      'Na_e_e':Nae_e, 'K_e_e':Ke_e, 'Cl_e_e':Cle_e,
                      'phi_i_e':phii_e, 'phi_e_e':phie_e, 'phi_m_e':phiM_e,
                      'I_M_e':JM_e, 'I_ch_Na':I_ch_Na, 'I_ch_K':I_ch_K,
                      'I_ch_Cl':I_ch_Cl}
        # source terms
        src_terms = {'f_Na_i':f_Nai, 'f_K_i':f_Ki, 'f_Cl_i':f_Cli,
                     'f_Na_e':f_Nae, 'f_K_e':f_Ke, 'f_Cl_e':f_Cle,
                     'f_phi_i':f_phii, 'f_phi_e':f_phie, 'f_I_M':f_JM,
                     'f_g_M':f_gM}
        # initial conditions
        init_conds = {'Na_i':init_Nai, 'K_i':init_Ki, 'Cl_i':init_Cli,
                      'Na_e':init_Nae, 'K_e':init_Ke, 'Cl_e':init_Cle,
                      'phi_m':init_phiM}
        # boundary terms
        bndry_terms = {'J_Na_e':J_Nae, 'J_K_e':J_Ke, 'J_Cl_e':J_Cle}

        return src_terms, exact_sols, init_conds, bndry_terms

# Class for converting symbolic SymPy functions to DOLFINx expressions
class SymPyToDOLFINxExpr():
    def __init__(self, x, time, sp_func, dim):
        self.time = time
        self.dim = dim
        self.f = lambdify([x[0], x[1], time], sp_func)
        
    def __call__(self, x):
        return self.f(x[0], x[1], self.time)

def mark_MMS_boundaries(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Mark internal and external facets of mesh for MMS test. """

    dim = mesh.topology.dim
    
    if dim == 2:
        facet_tags = mark_boundaries_square_MMS(mesh)
    elif dim == 3:
        facet_tags = mark_boundaries_cube_MMS(mesh)
    else:
        raise ValueError("Mesh dimension must be 2 or 3.")
    
    return facet_tags