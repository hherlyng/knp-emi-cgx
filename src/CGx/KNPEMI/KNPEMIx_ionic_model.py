import ufl
import time

import numpy   as np
import dolfinx as dfx

from abc      import ABC, abstractmethod
from mpi4py   import MPI
from petsc4py import PETSc

class IonicModel(ABC):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the base IonicModel class. 
        
        Parameters
        ----------
        KNPEMIx_problem : KNPEMIxProblem
            The KNPEMIx problem instance.
        
        tags : tuple, optional
            The membrane tags where the ionic model is applied.
            Default is None, in which case all membrane tags are used.
        
        """
        self.problem = KNPEMIx_problem	
        self.tags = tags

        # If tags are not specified, all the membrane tags are used
        if self.tags == None:
            self.tags = self.problem.gamma_tags

        # Transform int to tuple if needed
        if isinstance(self.tags, int): self.tags = (self.tags,)

        # Initialize a zero constant
        self.zero = dfx.fem.Constant(self.problem.mesh, 0.0)

    @abstractmethod
    def _init(self):
        # Abstract method that must be implemented by concrete subclasses.
        # Initialize membrane model-dependent quantities
        pass
        
    @abstractmethod
    def _eval(self, ion_idx):
        # Abstract method that must be implemented by concrete subclasses.
        pass

    def f_NKCC1(self,
                K_e: ufl.Coefficient,
                K_e_0: dfx.fem.Constant,
                K_min_val: float=3.0,
                eps: float=1e-6,
                cap: float=1.0) -> ufl.Coefficient:
        """ Function that silences the NKCC1 cotransporter at lowK_e and is zero when 
        K_e is not in the interval [K_min, K_e_0]."""
        # Define K_min as a dolfinx.fem.Constant
        K_min = dfx.fem.Constant(self.problem.mesh, K_min_val)

        # Zero outside the band [K_min, K_e_0]
        if ufl.conditional(ufl.Or(
                ufl.lt(K_e, K_min),
                ufl.gt(K_e, K_e_0)
                        ),
                True,
                False  
            ):
            return self.zero
        
        # Inside the band: safe denominator with epsilon and cap
        denom = ufl.max(K_e - K_e_0, eps)
        val = 1.0 / (1.0 + (0.03 / denom)**10)

        return ufl.min(ufl.max(val, self.zero), cap)

class PassiveModel(IonicModel):
    
    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the passive ionic model. """
        super().__init__(KNPEMIx_problem, tags)

    def _init(self):
        pass

    def __str__(self):
        return f"Passive model"

    def _eval(self, ion_idx: int) -> dfx.fem.Function:	
        """ Returns the membrane potential as ionic current. """
        return self.problem.phi_m_prev

class KirNaKPumpModel(IonicModel):

    # Potassium buffering parameters
    rho_pump_val = 1.1*1.12e-6	# Maximum pump rate (mol/m**2 s)
    P_Na_i_val = 10.0          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
    P_K_e_val  = 1.5         # [K+]e  threshold for Na+/K+ pump (mol/m^3)

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the Kir-Na pump model. 
            This model includes an inward-rectifying potassium (passive) current
            based on the Kir4.1 channel, combined with an Na/K/ATPase pump.

        Parameters
        ----------
        KNPEMIx_problem : KNPEMIxProblem
            The KNPEMIx problem instance.
            
        tags : tuple, optional
            The membrane tags where this ionic model is applied.
        """

        super().__init__(KNPEMIx_problem, tags)

        p = KNPEMIx_problem # Problem instance
        self.E_K_init = dfx.fem.Constant(p.mesh, p.psi.value*np.log(p.K_e_init.value/p.K_i_g_init.value)) # Initial potassium Nernst potential
        self.rho_pump = dfx.fem.Constant(p.mesh, self.rho_pump_val) # Maximum pump rate [mol/m**2 s]
        self.P_Na_i = dfx.fem.Constant(p.mesh, self.P_Na_i_val) # [Na+]i threshold for Na+/K+ pump [mol/m^3]
        self.P_K_e = dfx.fem.Constant(p.mesh, self.P_K_e_val) # [K+]e  threshold for Na+/K+ pump [mol/m^3]

    def __str__(self):
        return f'Na/K/ATPase pump with passive inward-rectifying K current'

    def _init(self):
        """ Initialize Kir-Na pump parameters. """

        p = self.problem # Problem instance

        c_Na_i = p.u_p[0][0] # ICS Na+ concentration at previous timestep [mol/m^3]
        c_K_e  = p.u_p[1][1] # ECS K+ concentration at previous timestep [mol/m^3]

        # Define the pump coefficient [mol/m^2 s]
        self.pump_coeff = (
                        (1.0 / (1.0 + (self.P_Na_i/c_Na_i)**(3/2)))
                      * (1.0 / (1.0 + self.P_K_e/c_K_e))
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
        p = self.problem # Problem instance
        ion: dict = p.ion_list[ion_idx] # Ion dictionary
        ue_p:  list[dfx.fem.Function] = p.u_p[1] # ECS concentrations at previous timestep [mol/m^3]
        phi_m: dfx.fem.Function = p.phi_m_prev # Membrane potential at previous timestep [V]
        F: dfx.fem.Constant = p.F # Faraday's constant [C/mol]
        z: dfx.fem.Constant = ion['z'] # Ion valence
            
        # Evaluate f_kir function depending on ion
        if ion['name'] == 'K':
            # Potassium has Kir-Na function
            # that reflects inward-rectifying behavior
            delta_phi: ufl.Coefficient = phi_m - ion['E'] # Kir-Na variable
            f_kir: ufl.Coefficient = self.f_Kir(p.K_e_init,
                                        ue_p[ion_idx],
                                        self.E_K_init,
                                        delta_phi,
                                        phi_m
                                        )

            I_ATP: ufl.Coefficient = -2*z*F*self.pump_coeff # ATP pump current density [A/m^2]

        else:
            # f_kir = 1 for Na and Cl
            f_kir = dfx.fem.Constant(p.mesh, 1.0)

            # ATP pump current [A/m^2]
            if ion['name'] == 'Na':
                I_ATP: ufl.Coefficient = 3*z*F*self.pump_coeff
            else:
                I_ATP = dfx.fem.Constant(p.mesh, 0.0)

        I_kir: ufl.Coefficient = f_kir*ion['g_leak_g']*(phi_m - ion['E']) # Kir-Na current density [A/m^2]
        
        # Total current is ATP pump current + Kir-Na current [A/m^2]
        I_pump_k_g: ufl.Coefficient = I_kir + I_ATP

        return I_pump_k_g # Ionic current [A/m^2]

    def f_Kir(self,
              K_e_init: dfx.fem.Constant,
              K_e: dfx.fem.Function,
              E_K_init: ufl.Coefficient,
              delta_phi: ufl.Coefficient,
              phi_m: dfx.fem.Function) -> ufl.Coefficient:
        """ Evaluate and return the Kir-Na function f_kir defined in Halnes et al. 2013. 
        
        Parameters
        ----------
        K_e_init : dfx.fem.Constant
            Initial extracellular potassium concentration [mol/m^3].
        K_e : dfx.fem.Function
            Extracellular potassium concentration [mol/m^3].
        E_K_init : ufl.Coefficient
            Initial potassium Nernst potential [V].
        delta_phi : ufl.Coefficient
            Kir-Na variable defined as delta_phi = phi_m - E_K [V].
        phi_m : dfx.fem.Function
            Membrane potential at previous timestep [V].
        
        """
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

        # Define maximum cotransporter strengths [A/m^2]
        g_KCC1 = 7e-2 # Maximum conductivity of KCC1 [S / m^2]
        self.S_KCC1 = dfx.fem.Constant(p.mesh, g_KCC1 * p.psi.value) # Maximum current density of KCC1 [A/m^2]

        g_NKCC1 = 2e-2 # Maximum conductivity of NKCC1 [S / m^2]
        self.S_NKCC1 = dfx.fem.Constant(p.mesh, g_NKCC1 * p.psi.value) # Maximum current density of NKCC1 [A/m^2]

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

        p = self.problem # Problem instance
        ion   = p.ion_list[ion_idx] # Ion dictionary
        c_Na_i = p.u_p[0][0] # ICS Na+ concentration at previous timestep [mol/m^3]
        c_Na_e = p.u_p[1][0] # ECS Na+ concentration at previous timestep [mol/m^3]
        c_K_i  = p.u_p[0][1] # ICS K+ concentration at previous timestep [mol/m^3]
        c_K_e  = p.u_p[1][1] # ECS K+ concentration at previous timestep [mol/m^3]
        c_Cl_i = p.u_p[0][2] # ICS Cl- concentration at previous timestep [mol/m^3]
        c_Cl_e = p.u_p[1][2] # ECS Cl- concentration at previous timestep [mol/m^3]
        c_K_e_0 = p.K_e_init # Initial ECS K+ concentration [mol/m^3]

        # Define the KCC1 cotransporter current density [A/m^2]
        I_KCC1 = (self.S_KCC1
                    * ufl.ln(
                        (c_K_i * c_Cl_i)
                            /
                        (c_K_e*c_Cl_e)
                    )
                )
        
        # Define the NKCC1 cotransporter current density [A/m^2]
        silence_factor = self.f_NKCC1(c_K_e, c_K_e_0) # Function that silences NKCC1 at low K_e
                                                      # and is zero when K_e outside of [K_min, K_e_0]
        I_NKCC1 = (
                    self.S_NKCC1 * silence_factor
                    * 
                    ufl.ln(
                        (c_Na_e * c_K_e * c_Cl_e**2)
                            /
                        (c_Na_i * c_K_i * c_Cl_i**2)
                        )
                    )

        # Return ionic current depending on ion type
        if ion["name"]=="Na":
            return -I_NKCC1
        elif ion["name"]=="K":
            return (-I_NKCC1 + I_KCC1)
        else:
            return (2*I_NKCC1 - I_KCC1)
        
class NeuronalCotransporters(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
         """ Constructor for the KCC2/NKCC1 neuronal cotransporter ionic model. """
         
         super().__init__(KNPEMIx_problem, tags)

    def __str__(self):
         return "KCC2/NKCC1 Cotransporters"

    def _init(self):
        """ Initialize neuronal cotransporter parameters. """	

        # Define maximum cotransporter strengths [A/m^2]
        self.S_KCC2 = dfx.fem.Constant(self.problem.mesh, 0.0068) # Maximum current density of KCC2 [A/m^2]
        self.S_NKCC1 = dfx.fem.Constant(self.problem.mesh, 0.0023) # Maximum current density of NKCC1 [A/m^2]

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

        p = self.problem # Problem instance
        ion    = p.ion_list[ion_idx] # Ion dictionary
        c_Na_i = p.u_p[0][0] # ICS Na+ concentration at previous timestep [mol/m^3]
        c_Na_e = p.u_p[1][0] # ECS Na+ concentration at previous timestep [mol/m^3]
        c_K_i  = p.u_p[0][1] # ICS K+  concentration at previous timestep [mol/m^3]
        c_K_e  = p.u_p[1][1] # ECS K+  concentration at previous timestep [mol/m^3]
        c_Cl_i = p.u_p[0][2] # ICS Cl- concentration at previous timestep [mol/m^3]
        c_Cl_e = p.u_p[1][2] # ECS Cl- concentration at previous timestep [mol/m^3]
        c_K_e_0 = p.K_e_init # Initial ECS K+ concentration [mol/m^3]

        # Define the KCC2 cotransporter current density [A/m^2]
        I_KCC2 = (self.S_KCC2
                    * ufl.ln(
                        (c_K_i * c_Cl_i)
                            /
                        (c_K_e*c_Cl_e)
                    )
                )
        
        # Define the NKCC1 cotransporter current density [A/m^2]
        silence_factor = self.f_NKCC1(c_K_e, c_K_e_0) # Function that silences NKCC1 at low K_e
                                                      # and is zero when K_e outside of [K_min, K_e_0]
        I_NKCC1 = (
                    self.S_NKCC1 * silence_factor
                    * 
                    ufl.ln(
                        (c_Na_e * c_K_e * c_Cl_e**2)
                            /
                        (c_Na_i * c_K_i * c_Cl_i**2)
                        )
                    )

        # Return ionic current depending on ion type
        if ion["name"]=="Na":
            return -I_NKCC1
        elif ion["name"]=="K":
            return (-I_NKCC1 + I_KCC2)
        else:
            return (I_NKCC1 - I_KCC2)
        
class ATPPump(IonicModel):

    def __init__(self, KNPEMIx_problem, tags: tuple=None):
        """ Constructor for the Na/K/ATPase pump ionic model. """

        super().__init__(KNPEMIx_problem, tags)

    def __str__(self):
        return "Na/K/ATPase pump"
    
    def _init(self):	
        """ Initialize neuronal ATP pump parameters. """

        # Define ATP pump parameters
        self.I_hat = dfx.fem.Constant(self.problem.mesh, 0.25) # Maximum pump strength [A/m^2]
        self.P_K_e  = dfx.fem.Constant(self.problem.mesh, 1.5)  # ECS K+  pump threshold [mM]
        self.P_Na_i = dfx.fem.Constant(self.problem.mesh, 10.0) # ICS Na+ pump threshold [mM]

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

        p   = self.problem # Problem instance
        ion = p.ion_list[ion_idx] # Ion dictionary

        if ion["name"]=="Cl":
            # Cl- is not affected by the ATP pump
            return dfx.fem.Constant(p.mesh, 0.0)

        c_Na_i = p.u_p[0][0] # ICS Na+ concentration at previous timestep [mol/m^3]
        c_K_e  = p.u_p[1][1] # ECS K+ concentration at previous timestep [mol/m^3]

        # Define the ATP pump current density [A/m^2]
        par_1 = 1 + self.P_K_e  / c_K_e
        par_2 = 1 + self.P_Na_i / c_Na_i
        I_ATP = self.I_hat / (par_1**2 * par_2**3) # ATP pump current [A/m^2]

        # Return ionic current depending on ion type
        if ion["name"]=="Na":
            return 3*I_ATP
        elif ion["name"]=="K":
            return -2*I_ATP
        else:
            raise ValueError("Unknown ion for ATP pump model.")

class HodgkinHuxley(IonicModel):

    def __init__(self, KNPEMIx_problem,
                 tags: tuple=None,
                 use_Rush_Larsen: bool=True,
                 time_steps_ODE: int=25):
        """ Constructor for the Hodgkin-Huxley ionic model. 
        
        Parameters
        ----------
        KNPEMIx_problem : KNPEMIxProblem
            The KNPEMIx problem instance.  

        tags : tuple, optional
            The membrane tags where this ionic model is applied.
            Default is None, in which case all membrane tags are used.
        
        use_Rush_Larsen : bool, optional
            Whether to use the Rush-Larsen method for updating gating variables.
            Default is True.
        
        time_steps_ODE : int, optional
            Number of ODE timesteps per PDE timestep. Default is 25.
        """
        
        super().__init__(KNPEMIx_problem, tags)

        self.use_Rush_Larsen = use_Rush_Larsen # Whether to use Rush-Larsen method for gating variables
        self.time_steps_ODE = time_steps_ODE # Number of ODE timesteps per PDE timestep
        self.T_stim = KNPEMIx_problem.T_stim.value # Stimulus period [s]

        if hasattr(KNPEMIx_problem, 'tau_syn_rise'):
            # Synaptic rise and decay time constants [s] 
            self.tau_syn_rise  = dfx.fem.Constant(KNPEMIx_problem.mesh, dfx.default_scalar_type(KNPEMIx_problem.tau_syn_rise))     
            self.tau_syn_decay = dfx.fem.Constant(KNPEMIx_problem.mesh, dfx.default_scalar_type(KNPEMIx_problem.tau_syn_decay))

    def __str__(self):
        return 'Hodgkin-Huxley'

    def _init(self):
        """ Initialize gating variables and time modulo variable used in the Hodgkin-Huxley model. """

        # Alias
        p = self.problem		

        # Gating variable finite element functions
        p.n = dfx.fem.Function(p.V); p.n.name = "n"
        p.m = dfx.fem.Function(p.V); p.m.name = "m"
        p.h = dfx.fem.Function(p.V); p.h.name = "h"

        # Set initial values for gating variables
        p.n.x.array[:] = p.n_init.value
        p.m.x.array[:] = p.m_init.value
        p.h.x.array[:] = p.h_init.value

        PETSc.Sys.Print(f"Initial n = {p.n_init.value}\nm = {p.m_init.value}\nh = {p.h_init.value}")

        # Set modulo time variable
        self.t_mod = dfx.fem.Constant(p.mesh, 0.0)

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
        p     = self.problem # Problem instance
        ion   = p.ion_list[ion_idx] # Ion dictionary

        # Get the leak conductivity for this ion
        g_k = ion['g_leak'] # Leak conductivity [S/m^2]

        # Define voltage-gated currents for sodium or potassium
        # No voltage-gated currents for chloride
        if ion['name'] == 'Na': 
            g_k += p.g_Na_bar*p.m**3*p.h # [S/m^2]
        elif ion['name'] == 'K':
            g_k += p.g_K_bar*p.n**4 # [S/m^2]
        
        # Return the total contribution to the ion current density [A/m^2]
        return g_k * (p.phi_m_prev - ion['E'])

    def _add_stimulus(self,
                    ion_idx: int,
                    step: bool,
                    range: list | np.ndarray=None,
                    dir: str=None
                ) -> ufl.Coefficient:
        """ Evaluate and return the stimulus part of the channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        step : bool
            If True, stimulus is introduced as a step function.
            If False, an exponential rise and decay stimulus is used.

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
        if step:
            # Introduce exponential decaying stimulus
            # as a Heaviside step function.
            exp_factor = ufl.exp(-self.t_mod / p.a_syn)
        else:
            # Introduce stimulus gradually with exponential rise and decay
            exp_factor = ufl.exp(-self.t_mod/self.tau_syn_decay) - ufl.exp(-self.t_mod/self.tau_syn_rise)

        if range is None:
            mask = 1.0
        else:
            # Create mask so that stimulus is zero outside of 
            # a subregion, but active within the subregion
            x = ufl.SpatialCoordinate(p.mesh)
            coord = x[dir]
            coord_min: float = range[0]
            coord_max: float = range[1]
            mask = ufl.conditional(
                        ufl.And(
                            ufl.gt(coord, coord_min), 
                            ufl.lt(coord, coord_max)
                            ), 
                        1.0, 0.0
                    )
        
        # Define the stimulus current [A/m^2]
        stim_current = mask * p.g_syn_bar * exp_factor * (p.phi_m_prev - ion["E"]) 

        if p.scale_stimulus:
            # Scale the stimulus current by the surface area of the stimulus tag
            p.stimulus_area = dfx.fem.assemble_scalar(
                                dfx.fem.form(
                                    mask * p.dS(p.stimulus_tags)
                                )
                            )
            p.stimulus_area = p.comm.allreduce(p.stimulus_area, op=MPI.SUM)
            PETSc.Sys.Print(f"Stimulus area on tag {p.stimulus_tags[0]}: {p.stimulus_area:0.6e} m^2")
            scale = 1.0 / p.stimulus_area
            stim_current *= scale

        return stim_current

    def update_gating_variables(self):
        """ Update the gating variables n, m, h using either the Rush-Larsen method or Forward Euler. """

        # Start timer
        tic = time.perf_counter()
        
        p = self.problem # Problem instance
        dt_ode = p.dt.value / self.time_steps_ODE # ODE timestep [s]

        # Compute V_M variable for the Hodgkin-Huxley equations
        # V_M = phi_m - phi_rest, measured in mV
        with p.phi_m_prev.x.petsc_vec.localForm() as loc_phi_m_prev:
            V_M = 1000*(loc_phi_m_prev[:] - p.phi_rest.value) # convert phi_m to mV	
        
        # Gating variable rate coefficients
        alpha_n = 0.01e3 * (10.-V_M) / (np.exp((10. - V_M)/10.) - 1.)
        beta_n  = 0.125e3 * np.exp(-V_M/80.)
        alpha_m = 0.1e3 * (25. - V_M) / (np.exp((25. - V_M)/10.) - 1)
        beta_m  = 4.e3 * np.exp(-V_M/18.)
        alpha_h = 0.07e3 * np.exp(-V_M/20.)
        beta_h  = 1.e3 / (np.exp((30. - V_M)/10.) + 1)

        if self.use_Rush_Larsen:
            # Precompute Rush-Larsen coefficients
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
            # Forward Euler: scale alpha and beta by dt
            alpha_n *= dt_ode
            beta_n  *= dt_ode
            alpha_m *= dt_ode
            beta_m  *= dt_ode
            alpha_h *= dt_ode
            beta_h  *= dt_ode
        
        # Solve the ODEs for the gating variables
        for _ in range(self.time_steps_ODE): 
            
            if self.use_Rush_Larsen:
                # Get vector entries local to process + ghosts
                with p.n.x.petsc_vec.localForm() as loc_n, p.m.x.petsc_vec.localForm() as loc_m, p.h.x.petsc_vec.localForm() as loc_h:

                    loc_n[:] = y_inf_n + (loc_n[:] - y_inf_n) * y_exp_n
                    loc_m[:] = y_inf_m + (loc_m[:] - y_inf_m) * y_exp_m
                    loc_h[:] = y_inf_h + (loc_h[:] - y_inf_h) * y_exp_h

            else:
                # Get vector entries local to process + ghosts
                with p.n.x.petsc_vec.localForm() as loc_n, p.m.x.petsc_vec.localForm() as loc_m, p.h.x.petsc_vec.localForm() as loc_h:

                    loc_n[:] += alpha_n * (1 - loc_n[:]) - beta_n * loc_n[:]
                    loc_m[:] += alpha_m * (1 - loc_m[:]) - beta_m * loc_m[:]
                    loc_h[:] += alpha_h * (1 - loc_h[:]) - beta_h * loc_h[:]	

        toc = time.perf_counter()
        ODE_step_time = p.mesh.comm.allreduce(toc-tic, op=MPI.MAX)
        PETSc.Sys.Print(f"ODE step in {ODE_step_time:0.4f} seconds")   	
    
    def update_t_mod(self, tol: float=1e-12):
        """ Update the modulo time variable used for synaptic stimuli. """
        self.t_mod.value = np.mod(self.problem.t.value + tol, self.T_stim)