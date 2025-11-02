import ufl
import time

import numpy   as np
import dolfinx as dfx

from abc      import ABC, abstractmethod
from mpi4py   import MPI
from petsc4py import PETSc

class IonicModel(ABC):

    def __init__(self, KNPEMIx_problem, tags=None):
        """ Constructor for the base IonicModel class. """
        self.problem = KNPEMIx_problem	
        self.tags = tags

        # If tags are not specified, all the membrane tags are used
        if self.tags == None:
            self.tags = self.problem.gamma_tags

        # Transform int to tuple if needed
        if isinstance(self.tags, int): self.tags = (self.tags,)

    @abstractmethod
    def _init(self):
        # Abstract method that must be implemented by concrete subclasses.
        # Initialize membrane model-dependent quantities
        pass
        
    @abstractmethod
    def _eval(self, ion_idx):
        # Abstract method that must be implemented by concrete subclasses.
        pass

class PassiveModel(IonicModel):
    
    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the passive ionic model. """
        super().__init__(KNPEMIx_problem, tags)

    def _init(self):
        pass

    def __str__(self):
        return f"Passive model"

    def _eval(self, ion_idx: int) -> ufl.Coefficient:	
        """ Returns the membrane potential as ionic current. """
        return self.problem.phi_m_prev

class KirNaKPumpModel(IonicModel):

    # Potassium buffering parameters
    rho_pump_val = 1.12e-6	# Maximum pump rate (mol/m**2 s)
    P_Na_i_val = 10.0          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
    P_K_e_val  = 1.5         # [K+]e  threshold for Na+/K+ pump (mol/m^3)
    k_dec_val = 2.9e-8		# Decay factor for [K+]e (m/s)

    # -k_dec * ([K]e − [K]e_0) both for K and Na
    use_decay_currents = False

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the Kir-Na pump ionic model. """

        super().__init__(KNPEMIx_problem, tags)

        # Initial potassium Nernst potential
        p = KNPEMIx_problem
        self.E_K_init = dfx.fem.Constant(p.mesh, p.psi.value*np.log(p.K_e_init.value/p.K_i_g_init.value))
        self.rho_pump = dfx.fem.Constant(p.mesh, self.rho_pump_val)
        self.P_Na_i = dfx.fem.Constant(p.mesh, self.P_Na_i_val)
        self.P_K_e = dfx.fem.Constant(p.mesh, self.P_K_e_val)
        if self.use_decay_currents:
            self.k_dec = dfx.fem.Constant(p.mesh, self.k_dec_val)

    def __str__(self):
        return f'Inward-rectifying K current passive model with Na/K/ATPase pump'

    def _init(self):
        """ Initialize Kir-Na pump parameters. """

        p = self.problem

        ui_p = p.u_p[0]
        ue_p = p.u_p[1]

        self.pump_coeff = (
                        1.0 / (1.0 + (self.P_Na_i/ui_p[0])**(3/2))
                     * (1.0 / (1.0 + self.P_K_e/ue_p[1]))
                     * self.rho_pump
                    )

    def _eval(self, ion_idx: int) -> ufl.Coefficient:
        """ Evaluate and return the ionic channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        I_ch_k : ufl.Coefficient
            The ionic channel current.
        """

        # Aliases		
        p = self.problem
        phi_m = p.phi_m_prev		
        ue_p  = p.u_p[1]
        F     = p.F
        ion   = p.ion_list[ion_idx]
        z     = ion['z']
            
        # Evaluate f_kir function depending on ion
        if ion['name'] == 'K':
            
            # Kir-Na variable		
            delta_phi  = phi_m - ion['E']
            f_kir = self.f_Kir(p.K_e_init, ue_p[ion_idx], self.E_K_init, delta_phi, phi_m)
            
            # ATP pump current
            I_ATP = -2*F*self.pump_coeff

        else:
            # Kir-Na variable
            f_kir = dfx.fem.Constant(p.mesh, 1.0)

            # ATP pump current
            if ion['name'] == 'Na':
                I_ATP = 3*F*self.pump_coeff
            else:
                I_ATP = 0.0

        I_ch_k = f_kir*ion['g_leak_g']*(phi_m - ion['E']) + I_ATP

        if self.use_decay_currents:
            if ion['name'] == 'K' or ion['name'] == 'Na':
                I_ch_k -= F*z*self.k_dec*(ue_p[1] - p.K_e_init)  
        
        return I_ch_k
    
    def f_Kir(self, K_e_init, K_e, E_K_init, delta_phi, phi_m) -> ufl.Coefficient:
        """ Evaluate and return the Kir-Na function f_kir defined in Halnes et al. 2013(?). """
        A = 1 + ufl.exp(0.433)
        B = 1 + ufl.exp(-(0.1186 + E_K_init) / 0.0441)
        C = 1 + ufl.exp((delta_phi + 0.0185) / 0.0425)
        D = 1 + ufl.exp(-(0.1186 + phi_m) / 0.0441)

        f = ufl.sqrt(K_e / K_e_init) * A*B / (C*D)

        return f

class GlialCotransporters(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
         """ Constructor for the KCC1/NKCC1 glial cotransporter ionic model. """
         
         super().__init__(KNPEMIx_problem, tags)

    def __str__(self):
         return "KCC1/NKCC1 Cotransporters"

    def _init(self):	
        """ Initialize glial cotransporter parameters. """
        
        p = self.problem

        # Maximum cotransporter strengths [A/m^2]
        g_KCC1 = 7e-1 # [S / m^2]
        self.S_KCC1 = dfx.fem.Constant(p.mesh, g_KCC1 * p.psi.value)
        
        g_NKCC1 = 2e-2 # [S / m^2]
        self.S_NKCC1 = dfx.fem.Constant(p.mesh, g_NKCC1 * p.psi.value)

    def _eval(self, ion_idx: int) -> ufl.Coefficient:
        """ Evaluate and return the ionic channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        ufl.Coefficient
            The ionic channel current.
        """

        p = self.problem
        ion   = p.ion_list[ion_idx]
        c_Na_i = p.u_p[0][0]
        c_Na_e = p.u_p[1][0]
        c_K_i = p.u_p[0][1]
        c_K_e = p.u_p[1][1]
        c_Cl_i = p.u_p[0][2]
        c_Cl_e = p.u_p[1][2]

        I_KCC1 = (self.S_KCC1
                    * ufl.ln(
                        (c_K_e * c_Cl_e)
                            /
                        (c_K_i*c_Cl_i)
                    )
                )
        I_NKCC1 = (
                    self.S_NKCC1 / (1 + ufl.exp(16. - c_K_e))
                    * 
                    ufl.ln(
                        (c_Na_e * c_K_e * c_Cl_e**2)
                            /
                        (c_Na_i * c_K_i * c_Cl_i**2)
                        )
                    )

        if ion["name"]=="Na":
            return -I_NKCC1
        elif ion["name"]=="K":
            return (-I_NKCC1 - I_KCC1)
        else:
            return (2*I_NKCC1 + I_KCC1)

class NeuronalCotransporters(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
         """ Constructor for the KCC2/NKCC1 neuronal cotransporter ionic model. """
         
         super().__init__(KNPEMIx_problem, tags)

    def __str__(self):
         return "KCC2/NKCC1 Cotransporters"

    def _init(self):
        """ Initialize neuronal cotransporter parameters. """	

        mesh = self.problem.mesh

        # Maximum cotransporter strengths [A/m^2]
        self.S_KCC2 = dfx.fem.Constant(mesh, 0.0034)
        self.S_NKCC1 = dfx.fem.Constant(mesh, 0.023)

    def _eval(self, ion_idx: int) -> ufl.Coefficient:
        """ Evaluate and return the ionic channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        ufl.Coefficient
            The ionic channel current.
        """

        p = self.problem
        ion    = p.ion_list[ion_idx]
        c_Na_i = p.u_p[0][0]
        c_Na_e = p.u_p[1][0]
        c_K_i  = p.u_p[0][1]
        c_K_e  = p.u_p[1][1]
        c_Cl_i = p.u_p[0][2]
        c_Cl_e = p.u_p[1][2]

        I_KCC2 = (self.S_KCC2
                    * ufl.ln(
                        (c_K_e * c_Cl_e)
                            /
                        (c_K_i*c_Cl_i)
                    )
                )
        I_NKCC1 = (
                    self.S_NKCC1 / (1 + ufl.exp(16. - c_K_e))
                    * 
                    ufl.ln(
                        (c_Na_e * c_K_e * c_Cl_e**2)
                            /
                        (c_Na_i * c_K_i * c_Cl_i**2)
                        )
                    )

        if ion["name"]=="Na":
            return -I_NKCC1
        elif ion["name"]=="K":
            return (-I_NKCC1 - I_KCC2)
        else:
            return (2*I_NKCC1 + I_KCC2)


class ATPPump(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the Na/K/ATPase pump ionic model. """

        super().__init__(KNPEMIx_problem, tags)

    def __str__(self):
        return "Na/K/ATPase pump"
    
    def _init(self):	
        """ Initialize neuronal ATP pump parameters. """

        mesh = self.problem.mesh

        self.I_hat = dfx.fem.Constant(mesh, 0.0449) # Maximum pump strength [A/m^2]
        self.m_K = dfx.fem.Constant(mesh, dfx.default_scalar_type(2.0)) # ECS K+ pump threshold [mM]
        self.m_Na = dfx.fem.Constant(mesh, dfx.default_scalar_type(7.7)) # ICS Na+ pump threshold [mM]

    def _eval(self, ion_idx: int) -> ufl.Coefficient:
        """ Evaluate and return the ionic channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        ufl.Coefficient
            The ionic channel current.
        """

        p = self.problem
        ion    = p.ion_list[ion_idx]
        c_Na_i = p.u_p[0][0]
        c_K_e  = p.u_p[1][1]

        par_1 = 1 + self.m_K  / c_K_e
        par_2 = 1 + self.m_Na / c_Na_i
        I_ATP = self.I_hat / (par_1**2 * par_2**3)

        if ion["name"]=="Na":
            return 3*I_ATP
        elif ion["name"]=="K":
            return -2*I_ATP        
        else:
            return dfx.fem.Constant(p.mesh, 0.0)

class HodgkinHuxley(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None,
                 use_Rush_Lar: bool=True, time_steps_ODE: int=25):
        """ Constructor for the Hodgkin-Huxley ionic model. """
        super().__init__(KNPEMIx_problem, tags)

        self.use_Rush_Lar = use_Rush_Lar
        self.time_steps_ODE = time_steps_ODE
        self.T = KNPEMIx_problem.T_stim.value # Stimulus period [s]

    def __str__(self):
        return 'Hodgkin-Huxley'

    def _init(self):
        """ Initialize gating variables and time modulo variable used in the Hodgkin-Huxley model. """

        # Alias
        p = self.problem		

        p.n = dfx.fem.Function(p.V); p.n.name = "n"
        p.m = dfx.fem.Function(p.V); p.m.name = "m"
        p.h = dfx.fem.Function(p.V); p.h.name = "h"

        p.n.x.array[:] = p.n_init.value
        p.m.x.array[:] = p.m_init.value
        p.h.x.array[:] = p.h_init.value

        PETSc.Sys.Print("Initial n, m, h:", p.n_init.value, p.m_init.value, p.h_init.value)

        # Set modulo time variable
        p.t_mod = dfx.fem.Constant(p.mesh, 0.0)

    def _eval(self, ion_idx: int) -> ufl.Coefficient:
        """ Evaluate and return the (passive) ionic channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        ufl.Coefficient
            The ionic channel current.
        """
        # Aliases		
        p     = self.problem		
        ion   = p.ion_list[ion_idx]

        # Add leak conductivity to the channel conductivity
        g_k = ion['g_leak']

        # Voltage-gated currents
        if ion['name'] == 'Na': 
            g_k += p.g_Na_bar*p.m**3*p.h

        elif ion['name'] == 'K':
            g_k += p.g_K_bar*p.n**4

        # Return the total contribution to the channel current
        return g_k * (p.phi_m_prev - ion['E'])

    def _add_stimulus(self,
                    ion_idx: int,
                    range: list | np.ndarray=None,
                    dir: str=None
                ) -> ufl.Coefficient:
        """ Evaluate and return the stimulus part of the channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        type float
            The stimulus part of the channel current.
        """

        # Aliases
        p = self.problem
        ion = p.ion_list[ion_idx]

        assert ion["name"] == "Na", print(
            "Only Na can have a stimulus current in the Hodgkin-Huxley model."
        )

        # Synaptic conductivity factor
        g_syn_fac = p.g_syn_bar * ufl.exp(-p.t_mod / p.a_syn)

        if range is None:
            mask = 1.0
        else:
            # Create mask so that stimulus is zero outside of 
            # a subregion, but active within the subregion
            x = ufl.SpatialCoordinate(p.mesh)
            coord = x[dir]
            coord_min = range[0]
            coord_max = range[1]
            mask = ufl.conditional(ufl.And(ufl.gt(coord, coord_min), ufl.lt(coord, coord_max)), 1.0, 0.0)

        # Return stimulus
        return mask * g_syn_fac * (p.phi_m_prev - ion["E"])

    def update_gating_variables(self):
        """ Update the gating variables n, m, h using either the Rush-Larsen method or Forward Euler. """

        tic = time.perf_counter()

        # aliases	
        n = self.problem.n
        m = self.problem.m
        h = self.problem.h
        phi_m_prev = self.problem.phi_m_prev
        dt_ode     = float(self.problem.dt.value) / self.time_steps_ODE

        # Set membrane potential
        with phi_m_prev.x.petsc_vec.localForm() as loc_phi_m_prev:
            V_M = 1000*(loc_phi_m_prev[:] - self.problem.phi_rest.value) # convert phi_m to mV	
        
        alpha_n = 0.01e3 * (10.-V_M) / (np.exp((10. - V_M)/10.) - 1.)
        beta_n  = 0.125e3 * np.exp(-V_M/80.)
        alpha_m = 0.1e3 * (25. - V_M) / (np.exp((25. - V_M)/10.) - 1)
        beta_m  = 4.e3 * np.exp(-V_M/18.)
        alpha_h = 0.07e3 * np.exp(-V_M/20.)
        beta_h  = 1.e3 / (np.exp((30. - V_M)/10.) + 1)

        if self.use_Rush_Lar:
            
            tau_y_n = 1.0/(alpha_n + beta_n)
            tau_y_m = 1.0/(alpha_m + beta_m)
            tau_y_h = 1.0/(alpha_h + beta_h)

            y_inf_n = alpha_n * tau_y_n
            y_inf_m = alpha_m * tau_y_m
            y_inf_h = alpha_h * tau_y_h

            y_exp_n =  np.exp(-dt_ode/tau_y_n)
            y_exp_m =  np.exp(-dt_ode/tau_y_m)
            y_exp_h =  np.exp(-dt_ode/tau_y_h)
            
        else:

            alpha_n *= dt_ode
            beta_n  *= dt_ode
            alpha_m *= dt_ode
            beta_m  *= dt_ode
            alpha_h *= dt_ode
            beta_h  *= dt_ode
        
        for _ in range(self.time_steps_ODE): 
            
            if self.use_Rush_Lar:
                # Get vector entries local to process + ghosts
                with n.x.petsc_vec.localForm() as loc_n, m.x.petsc_vec.localForm() as loc_m, h.x.petsc_vec.localForm() as loc_h:

                    loc_n[:] = y_inf_n + (loc_n[:] - y_inf_n) * y_exp_n
                    loc_m[:] = y_inf_m + (loc_m[:] - y_inf_m) * y_exp_m
                    loc_h[:] = y_inf_h + (loc_h[:] - y_inf_h) * y_exp_h

            else:
                # Get vector entries local to process + ghosts
                with n.x.petsc_vec.localForm() as loc_n, m.x.petsc_vec.localForm() as loc_m, h.x.petsc_vec.localForm() as loc_h:

                    loc_n[:] += alpha_n * (1 - loc_n[:]) - beta_n * loc_n[:]
                    loc_m[:] += alpha_m * (1 - loc_m[:]) - beta_m * loc_m[:]
                    loc_h[:] += alpha_h * (1 - loc_h[:]) - beta_h * loc_h[:]	

        toc = time.perf_counter()
        ODE_step_time = self.problem.mesh.comm.allreduce(toc-tic, op=MPI.MAX)
        PETSc.Sys.Print(f"ODE step in {ODE_step_time:0.4f} seconds")   	
    
    def update_t_mod(self):
        """ Update the modulo time variable used for synaptic stimulus. """
        self.problem.t_mod.value = np.mod(self.problem.t.value + 1e-10, self.T) # Add a small number to t to avoid floating-point precision problems
