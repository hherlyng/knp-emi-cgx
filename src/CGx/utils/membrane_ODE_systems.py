from abc import abstractmethod, ABC
from petsc4py import PETSc
import numpy   as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

print = PETSc.Sys.Print

class MembraneODESystem(ABC):
    """ Class to solve the ODE system for initial conditions of membrane potential and ionic concentrations. """
    def __init__(self,
                problem,
                plot_show: bool=False,
                plot_save: bool=False,
                stimulus_flag: bool=False,
                timestep: float=1e-3,
                max_time: float=500.0,
                verbose: bool=False):
        """ Initialize the ODE system.

            Parameters
            ----------
            problem : KNPEMIxProblem
                The KNPEMIxProblem instance containing problem parameters
            plot_show : bool, optional
                Whether to enable plotting of results, by default False
            plot_save : bool, optional
                Whether to save plots of results, by default False
            stimulus_flag : bool, optional
                Whether to include synaptic stimulus in the model, by default False
            timestep : float, optional
                The time step for the ODE solver, by default 1e-6
            max_time : float, optional
                The maximum simulation time, by default 3.0
            verbose : bool, optional
                Whether to enable verbose output, by default False
        """
        # Store parameters
        self.problem = problem
        self.plot = plot_show or plot_save
        self.plot_show = plot_show
        self.plot_save = plot_save
        self.stimulus = stimulus_flag
        self.timestep = timestep
        self.max_time = max_time
        self.verbose = verbose

        # Define timespan for ODE solver
        num_timesteps = int(max_time / timestep)
        self.times = np.linspace(0, max_time, num_timesteps+1)

        self.initialize_constants()
    
    @abstractmethod
    def initialize_constants(self):
        """ Initialize constants and initial guesses for the ODE system. """
        pass
    
    @abstractmethod
    def initialize_initial_conditions(self, init_cond_array: list[float]):
        """ Initialize the initial conditions from an array.

            Parameters
            ----------
            init_cond_array : list[float]
                Array of initial conditions for the ODE system
        """
        pass

    @abstractmethod
    def solve_ode_system(self) -> list[float]:
        """ Solve the ODE system to steady state.

            Returns
            -------
            list[float]
                The steady-state solution of the ODE system
        """
        pass
    
    @abstractmethod
    def initialize_plotting(self):
        """ Initialize arrays for plotting results. """
        pass

    @abstractmethod
    def append_arrays(self, sol_: list[float]):
        """ Append current solution to plotting arrays.

            Parameters
            ----------
            sol_ : list[float]
                Current solution vector
        """
        pass

    @abstractmethod
    def plot_results(self):
        """ Plot the results of the ODE system solution. """
        pass


    # Cotransporter currents
    def f_NKCC1(self, K_e: float, K_e_0: float, K_min: float=3.0, eps: float=1e-6, cap: float=1.0) -> float:
        """ Function that silences the NKCC1 cotransporter at low K_e and is zero when 
        K_e is not in the interval [K_min, K_e_0]."""
        # Zero outside the band [K_min, K_e_0]
        if K_e <= K_min or K_e >= K_e_0:
            return 0.0
        
        # Inside the band: safe denominator with epsilon and cap
        denom = max(K_e - K_e_0, eps)
        val = 1.0 / (1.0 + (0.03 / denom)**10)

        return min(max(val, 0.0), cap)


class ThreeCompartmentMembraneODESystem(MembraneODESystem):
    """ Class to solve the ODE system for a three-compartment model (neuron + glia + ECS). """

    def initialize_constants(self):
        """ Initialize constants and initial guesses for the ODE system. """
        # Initialize plotting arrays if enabled
        if self.plot:
            self.initialize_plotting()

        # Initialize constants and initial guesses
        # from the problem instance
        p = self.problem # For brevity
        self.R = p.R.value # Gas constant [J/(mol*K)]
        self.F = p.F.value # Faraday's constant [C/mol]
        self.T = p.T.value # Temperature [K]
        self.C_M = p.C_M.value # Membrane capacitance [F/m**2]
        self.g_Na_bar = p.g_Na_bar.value # Na max conductivity [S/m**2]
        self.g_K_bar = p.g_K_bar.value # K max conductivity [S/m**2]
        self.g_Na_leak = p.g_Na_leak.value # Na leak conductivity [S/m**2]
        self.g_Na_leak_g = p.g_Na_leak_g.value # Na leak conductivity glia [S/m**2]
        self.g_K_leak = p.g_K_leak.value # K leak conductivity [S/m**2]
        self.g_K_leak_g = p.g_K_leak_g.value # K leak conductivity glia [S/m**2]
        self.g_Cl_leak = p.g_Cl_leak.value # Cl leak conductivity [S/m**2]
        self.g_Cl_leak_g = p.g_Cl_leak_g.value # Cl leak conductivity glia [S/m**2]
        self.phi_rest = p.phi_rest.value # Resting potential [V]
        self.phi_m_init = p.phi_m_init.value # Initial neuronal membrane potential [V]
        self.Na_i_init = p.Na_i_init.value # Initial neuronal intracellular Na+ concentration [mM]
        self.Na_e_init = p.Na_e_init.value # Initial extracellular Na+ concentration [mM]
        self.K_i_init = p.K_i_init.value # Initial neuronal intracellular K+ concentration [mM]
        self.K_e_init = p.K_e_init.value # Initial extracellular K+ concentration [mM]
        self.Cl_i_init = p.Cl_i_init.value # Initial neuronal intracellular Cl- concentration [mM]
        self.Cl_e_init = p.Cl_e_init.value # Initial extracellular Cl- concentration [mM]
        self.phi_m_g_init = p.phi_m_g_init.value # Initial glial membrane potential [V]
        self.Na_i_g_init = p.Na_i_g_init.value # Initial glial intracellular Na+ concentration [mM]
        self.K_i_g_init = p.K_i_g_init.value # Initial glial intracellular K+ concentration [mM]
        self.Cl_i_g_init = p.Cl_i_g_init.value # Initial glial intracellular Cl- concentration [mM]

        if self.stimulus:
            # Stimulus parameters
            g_syn_bar = p.g_syn_bar.value # Max synaptic conductance [S/m^2]
            T = p.T_stim.value # Stimulus period [s]
            a_syn = p.a_syn.value # Synaptic time constant [s]
            self.g_syn_fac = lambda t: g_syn_bar * np.exp((-np.mod(t+1e-10, T)) / a_syn)

    def initialize_initial_conditions(self, init_cond_array: list[float]):
        """ Initialize the initial conditions from an array.

            Parameters
            ----------
            init_cond_array : list[float]
                Array of initial conditions for the ODE system
        """
        (
            self.phi_m_init,
            self.Na_i_init,
            self.Na_e_init,
            self.K_i_init,
            self.K_e_init,
            self.Cl_i_init,
            self.Cl_e_init,
            self.phi_m_g_init,
            self.Na_i_g_init,
            self.K_i_g_init,
            self.Cl_i_g_init,
            self.n_init,
            self.m_init,
            self.h_init
        ) = init_cond_array
        
    def solve_ode_system(self) -> list[float]:

        p = self.problem # For brevity
        # Get constants
        R = self.R # Gas constant [J/(mol*K)]
        F = self.F # Faraday's constant [C/mol]
        T = self.T # Temperature [K]
        C_m = self.C_M # Membrane capacitance
        z_Na =  1 # Valence sodium
        z_K  =  1 # Valence potassium
        z_Cl = -1 # Valence chloride
        g_Na_stim_func = self.g_syn_fac if self.stimulus else 0.0
        g_Na_bar  = self.g_Na_bar                 # Na max conductivity (S/m**2)
        g_K_bar   = self.g_K_bar                  # K max conductivity (S/m**2)    
        g_Na_leak = self.g_Na_leak              # Na leak conductivity (S/m**2) (Constant)
        g_Na_leak_g = self.g_Na_leak_g              # Na leak conductivity (S/m**2) (Constant)
        g_K_leak  = self.g_K_leak              # K leak conductivity (S/m**2)
        g_K_leak_g  = self.g_K_leak_g              # K leak conductivity (S/m**2)
        g_Cl_leak = self.g_Cl_leak                  # Cl leak conductivity (S/m**2) (Constant)
        g_Cl_leak_g = self.g_Cl_leak_g # Cl leak conductivity (S/m**2) (Constant)
        phi_rest = self.phi_rest  # Resting potential [V]

        # Define initial condition guesses
        Na_i_0 = self.Na_i_init # [Mm]
        Na_e_0 = self.Na_e_init # [Mm]
        K_i_0 = self.K_i_init # [Mm]
        K_e_0 = self.K_e_init # [Mm]
        Cl_i_0 = self.Cl_i_init # [Mm]
        Cl_e_0 = self.Cl_e_init # [Mm]
        phi_m_0 = self.phi_m_init # [V]

        # ATP pump
        I_hat = 0.25 # Maximum pump strength [A/m^2]
        P_Na_i = 10          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
        P_K_e  = 1.5         # [K+]e  threshold for Na+/K+ pump (mol/m^3)

        # Cotransporters
        S_KCC2 = 0.0068
        S_NKCC1 = 0.00023

        # Hodgkin-Huxley parameters
        alpha_n = lambda V_m: 0.01e3 * (10.-V_m) / (np.exp((10. - V_m)/10.) - 1.)
        beta_n  = lambda V_m: 0.125e3 * np.exp(-V_m/80.)
        alpha_m = lambda V_m: 0.1e3 * (25. - V_m) / (np.exp((25. - V_m)/10.) - 1)
        beta_m  = lambda V_m: 4.e3 * np.exp(-V_m/18.)
        alpha_h = lambda V_m: 0.07e3 * np.exp(-V_m/20.)
        beta_h  = lambda V_m: 1.e3 / (np.exp((30. - V_m)/10.) + 1)

        # Set steady-state gating variables as initial guess
        V_m_gating = (phi_m_0 - phi_rest)*1e3 # Relative potential with unit correction
        n_0 = alpha_n(V_m_gating) / (alpha_n(V_m_gating) + beta_n(V_m_gating))
        m_0 = alpha_m(V_m_gating) / (alpha_m(V_m_gating) + beta_m(V_m_gating))
        h_0 = alpha_h(V_m_gating) / (alpha_h(V_m_gating) + beta_h(V_m_gating))

        # Nernst potential
        E = lambda z_k, c_ki, c_ke: R*T/(z_k*F) * np.log(c_ke/c_ki)

        # ATP current
        par_1 = lambda K_e: 1 + P_K_e / K_e
        par_2 = lambda Na_i: 1 + P_Na_i / Na_i
        I_ATP = lambda Na_i, K_e: \
                    I_hat / (par_1(K_e)**2 * par_2(Na_i)**3)

        # Cotransporter currents
        I_KCC2 = lambda K_i, K_e, Cl_i, Cl_e: \
                    S_KCC2 * np.log((K_i * Cl_i)/(K_e*Cl_e))
        I_NKCC1_n = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: \
                        S_NKCC1 * self.f_NKCC1(K_e, K_e_0) * np.log((Na_e * K_e * Cl_e**2)/(Na_i * K_i * Cl_i**2))


        # Volumes and surface areas in m^3 and m^2
        # Both neuronal and glial intracellular space
        vol_i_n = p.vol_i_n # [m^3]
        vol_i_g = p.vol_i_g # [m^3]
        vol_e = p.vol_e # [m^3]
        area_g_n = p.area_g_n # [m^2]
        area_g_g = p.area_g_g # [m^2]
        
        # Membrane potential initial conditions
        phi_m_0_n = phi_m_0 # Neuronal
        phi_m_0_g = self.phi_m_g_init # Glial [V]
        Na_i_0_g = self.Na_i_g_init # [Mm]
        K_i_0_g = self.K_i_g_init # [Mm]
        Cl_i_0_g = self.Cl_i_g_init # [Mm]

        # Glial mechanisms
        # Kir-Na and Na/K pump mechanisms
        E_K_0 = E(z_K, K_i_0, K_e_0)
        A = 1 + np.exp(0.433)
        B = 1 + np.exp(-(0.1186 + E_K_0) / 0.0441)
        C = lambda delta_phi_K: 1 + np.exp((delta_phi_K + 0.0185)/0.0425)
        D = lambda phi_m: 1 + np.exp(-(0.1186 + phi_m)/0.0441)

        rho_pump = 1.1*1.12e-6	 # Maximum pump rate (mol/m**2 s)

        # Pump expression
        I_glia_pump = lambda Na_i, K_e: rho_pump*F * (1 / (1 + (P_Na_i/Na_i)**(3/2))) * (1 / (1 + P_K_e/K_e))

        # Inward-rectifying K channel function
        f_Kir = lambda K_e, delta_phi_K, phi_m: A*B/(C(delta_phi_K)*D(phi_m))*np.sqrt(K_e/(K_e_0))

        # Cotransporter strength and current
        g_KCC1 = 7e-2 # [S / m^2]
        S_KCC1 = g_KCC1 * R*T / F
        I_KCC1 = lambda K_i, K_e, Cl_i, Cl_e: S_KCC1 * np.log((K_i * Cl_i) / (K_e * Cl_e))

        g_NKCC1_g = 2e-2 # [S / m^2]
        S_NKCC1_g = g_NKCC1_g * R*T / F
        I_NKCC1_g = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: S_NKCC1_g * self.f_NKCC1(K_e, K_e_0) * np.log((Na_e * K_e * Cl_e**2)/(Na_i * K_i * Cl_i**2))
        
        # Define right-hand side of ODE system
        def three_compartment_rhs(t: float, x: list[float]) -> list[float]:
            """ Right-hand side of ODE system for three-compartment system (neuron + glia + ECS). 
            
                Parameters
                ----------
                t : float
                    Current time [s]
                x : list[float]
                    Current state vector
            """
            # Extract variables at previous timestep
            phi_m_n_ = x[0]; Na_i_n_ = x[1]; Na_e_ = x[2]; K_i_n_ = x[3]; K_e_ = x[4]; Cl_i_n_ = x[5]; Cl_e_ = x[6]
            phi_m_g_ = x[7]; Na_i_g_ = x[8]; K_i_g_ = x[9]; Cl_i_g_ = x[10]; n = x[11]; m = x[12]; h = x[13]

            # Neuronal mechanisms
            # Define potential used in gating variable expressions
            phi_m_gating = (phi_m_n_ - phi_rest)*1e3 # Relative potential with unit correction

            # Calculate Nernst potentials
            E_Na_n = E(z_Na, Na_i_n_, Na_e_)
            E_K_n  = E(z_K, K_i_n_, K_e_)
            E_Cl_n = E(z_Cl, Cl_i_n_, Cl_e_)

            g_Na_stim = g_Na_stim_func(t) if self.stimulus else 0.0
            
            I_ATP_n_ = I_ATP(Na_i_n_, K_e_)
            I_NKCC1_n_ = I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
            I_KCC2_ = I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)

            # Calculate neuronal ionic currents
            I_Na_n = (
                    (g_Na_leak + g_Na_bar * m**3 * h) * (phi_m_n_ - E_Na_n)
                    + 3*I_ATP_n_
                    - I_NKCC1_n_
                )
            I_K_n = (
                    (g_K_leak + g_K_bar * n**4)* (phi_m_n_ - E_K_n)
                    - 2*I_ATP_n_
                    - I_NKCC1_n_
                    + I_KCC2_
                )
            I_Cl_n = (
                    g_Cl_leak * (phi_m_n_ - E_Cl_n)
                    + 2*I_NKCC1_n_
                    - I_KCC2_
                )
            # Total neuronal ionic current [A/m^2]
            I_ion_n = I_Na_n + I_K_n + I_Cl_n 
            
            # Glial mechanisms
            # Calculate Nernst potentials [V]
            E_Na_g = E(z_Na, Na_i_g_, Na_e_)
            E_K_g  = E(z_K, K_i_g_, K_e_)
            E_Cl_g = E(z_Cl, Cl_i_g_, Cl_e_)
            
            # Calculate glial ionic currents
            delta_phi_K = phi_m_g_ - E_K_g
            I_ATP_g_ = I_glia_pump(Na_i_g_, K_e_)
            I_NKCC1_g_ = I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_)
            I_KCC1_ = I_KCC1(K_i_g_, K_e_, Cl_i_g_, Cl_e_)
            
            I_Na_g = (
                    g_Na_leak_g * (phi_m_g_ - E_Na_g)
                    + 3*I_ATP_g_
                    - I_NKCC1_g_
                )
            I_K_g = (
                    g_K_leak_g * f_Kir(K_e_, delta_phi_K, phi_m_g_) * (phi_m_g_ - E_K_g)
                    - 2*I_ATP_g_
                    - I_NKCC1_g_
                    + I_KCC1_
                )
            I_Cl_g = (
                    g_Cl_leak_g * (phi_m_g_ - E_Cl_g)
                    + 2*I_NKCC1_g_
                    - I_KCC1_
                )
            # Total glial ionic current density [A/m^2]
            I_ion_g = I_Na_g + I_K_g + I_Cl_g

            # Define right-hand expressions
            rhs_phi_n = -1/C_m * I_ion_n
            rhs_Na_i_n = -I_Na_n/(z_Na*F) * area_g_n / vol_i_n
            rhs_Na_e_n =  I_Na_n/(z_Na*F) * area_g_n / vol_e
            rhs_K_i_n = -I_K_n/(z_K*F) * area_g_n / vol_i_n
            rhs_K_e_n =  I_K_n/(z_K*F) * area_g_n / vol_e
            rhs_Cl_i_n = -I_Cl_n/(z_Cl*F) * area_g_n / vol_i_n
            rhs_Cl_e_n =  I_Cl_n/(z_Cl*F) * area_g_n / vol_e
            rhs_phi_g = -1/C_m * I_ion_g
            rhs_Na_i_g = -I_Na_g/(z_Na*F) * area_g_g / vol_i_g
            rhs_Na_e_g =  I_Na_g/(z_Na*F) * area_g_g / vol_e
            rhs_K_i_g = -I_K_g/(z_K*F) * area_g_g / vol_i_g
            rhs_K_e_g =  I_K_g/(z_K*F) * area_g_g / vol_e
            rhs_Cl_i_g = -I_Cl_g/(z_Cl*F) * area_g_g / vol_i_g
            rhs_Cl_e_g =  I_Cl_g/(z_Cl*F) * area_g_g / vol_e
            rhs_Na_e = rhs_Na_e_n + rhs_Na_e_g
            rhs_K_e  = rhs_K_e_n + rhs_K_e_g
            rhs_Cl_e = rhs_Cl_e_n + rhs_Cl_e_g
            rhs_n = alpha_n(phi_m_gating) * (1 - n) - beta_n(phi_m_gating) * n
            rhs_m = alpha_m(phi_m_gating) * (1 - m) - beta_m(phi_m_gating) * m
            rhs_h = alpha_h(phi_m_gating) * (1 - h) - beta_h(phi_m_gating) * h

            return [
                rhs_phi_n, rhs_Na_i_n, rhs_Na_e, rhs_K_i_n, rhs_K_e, rhs_Cl_i_n, rhs_Cl_e, # Neuronal variables
                rhs_phi_g, rhs_Na_i_g, rhs_K_i_g, rhs_Cl_i_g, # Glial variables
                rhs_n, rhs_m, rhs_h # Gating variables
                ]

        init = [
            phi_m_0_n, Na_i_0, Na_e_0, K_i_0, K_e_0, Cl_i_0, Cl_e_0,
            phi_m_0_g, Na_i_0_g, K_i_0_g, Cl_i_0_g, n_0, m_0, h_0,
                ]
        sol_ = init

        # Add initial condition to plotting arrays
        if self.plot: self.append_arrays(sol_)

        # Loop over timesteps and solve the ODE system
        for t, dt in zip(self.times, np.diff(self.times)):
        
            if t > 0:
                # Initial condition for ODE solver is
                # solution at previous timestep
                init = sol_

            # Integrate ODE system
            sol = solve_ivp(
                    lambda t, x: three_compartment_rhs(t, x),
                    [t, t+dt],
                    init,
                    method='Radau',
                    rtol=1e-6,
                    atol=1e-8
                )
            
            # Update previous solution
            sol_: list[float] = sol.y[:, -1] 

            if self.plot: self.append_arrays(sol_)
            if self.verbose:
                print(f"Time: {t:.6f} s")
                print(f"Solution = {sol_}")
                print(f"RHS = {three_compartment_rhs(t, sol_)}")

                print("-------- Currents --------")
                print(f"I_ATP_n = {I_ATP(sol_[1], sol_[4]):.6e} A/m^2")
                print(f"I_NKCC1_n = {I_NKCC1_n(sol_[1], sol_[2], sol_[3], sol_[4], sol_[5], sol_[6]):.6e} A/m^2")
                print(f"I_KCC2 = {I_KCC2(sol_[3], sol_[4], sol_[5], sol_[6]):.6e} A/m^2")
                print(f"I_ATP_g = {I_glia_pump(sol_[8], sol_[4]):.6e} A/m^2")
                print(f"I_NKCC1_g = {I_NKCC1_g(sol_[8], sol_[2], sol_[9], sol_[4], sol_[10], sol_[6]):.6e} A/m^2")
                print(f"I_KCC1 = {I_KCC1(sol_[9], sol_[4], sol_[10], sol_[6]):.6e} A/m^2")
                print(f"g_Kir = {g_K_leak_g * f_Kir(sol_[4], sol_[7] - E(z_K, sol_[9], sol_[4]), sol_[7]):.6e} A/m^2")
                print(f"I_Cl_leak_g = {g_Cl_leak_g*(sol_[7] - E(z_Cl, sol_[10], sol_[6])):.6e} A/m^2")
                print(f"I_Cl_leak_n = {g_Cl_leak*(sol_[0] - E(z_Cl, sol_[5], sol_[6])):.6e} A/m^2")
                print(f"I_Na_leak_n = {(g_Na_leak + g_Na_bar * sol_[12]**3 * sol_[13])*(sol_[0] - E(z_Na, sol_[1], sol_[2])):.6e} A/m^2")
                print(f"I_K_leak_n = {(g_K_leak + g_K_bar * sol_[11]**4)*(sol_[0] - E(z_K, sol_[3], sol_[4])):.6e} A/m^2")
                print(f"I_Na_leak_g = {g_Na_leak_g*(sol_[7] - E(z_Na, sol_[8], sol_[2])):.6e} A/m^2")
                print(f"I_K_leak_g = {g_K_leak_g * f_Kir(sol_[4], sol_[7] - E(z_K, sol_[9], sol_[4]), sol_[7])*(sol_[7] - E(z_K, sol_[9], sol_[4])):.6e} A/m^2")
                print("--------------------------------------------------")

            if np.allclose(three_compartment_rhs(t, sol_), 0.0, rtol=1e-8, atol=1e-10):
                print("Steady state reached. Derivatives zero to within tolerance.")
                [print(f"Variable {j}: {sol_[j]:.18f}") for j in range(len(sol_))]
                break

            # Checks
            if np.isclose(t+dt, self.max_time):
                print("Max time exceeded without finding steady state. Exiting.")
                break

            if any(np.isnan(sol_)):
                print("NaN values in solution. Exiting.")
                break
        
        if self.plot:
            self.plot_results()

        return sol_
    
    def initialize_plotting(self):
        """ Initialize arrays for plotting results. """
        
        self.phi_m_n_arr = []
        self.Na_i_n_arr = []
        self.Na_e_arr = []
        self.K_i_n_arr = []
        self.K_e_arr = []
        self.Cl_i_n_arr = []
        self.Cl_e_arr = []
        self.phi_m_g_arr = []
        self.Na_i_g_arr = []
        self.K_i_g_arr = []
        self.Cl_i_g_arr = []
        self.n_arr = []
        self.m_arr = []
        self.h_arr = []

    def append_arrays(self, sol_: list[float]):
        """ Append current solution to plotting arrays.

            Parameters
            ----------
            sol_ : list[float]
                Current solution vector
        """
        self.phi_m_n_arr.append(sol_[0])
        self.Na_i_n_arr.append(sol_[1])
        self.Na_e_arr.append(sol_[2])
        self.K_i_n_arr.append(sol_[3])
        self.K_e_arr.append(sol_[4])
        self.Cl_i_n_arr.append(sol_[5])
        self.Cl_e_arr.append(sol_[6])
        self.phi_m_g_arr.append(sol_[7])
        self.Na_i_g_arr.append(sol_[8])
        self.K_i_g_arr.append(sol_[9])
        self.Cl_i_g_arr.append(sol_[10])
        self.n_arr.append(sol_[11])
        self.m_arr.append(sol_[12])
        self.h_arr.append(sol_[13])

    def plot_results(self):
        """ Plot the results of the ODE system solution. 
        
            Parameters
            ----------
            show : bool, optional
                Whether to display the plots, by default False
            save : bool, optional
                Whether to save the plots as PNG files, by default False
        """
        times = np.arange(len(self.phi_m_n_arr)) * 1e-3 * self.timestep # Milliseconds
        figsize = (10, 6)

        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(times, self.phi_m_n_arr, label='Neuron Membrane Potential')
        ax1.plot(times, self.phi_m_g_arr, label='Glia Membrane Potential')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Membrane Potential [V]')
        ax1.legend()
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.plot(times, self.Na_i_n_arr, label='Neuron Na+ Intracellular')
        ax2.plot(times, self.Na_i_g_arr, label='Glia Na+ Intracellular')
        ax2.plot(times, self.Na_e_arr, label='Extracellular Na+')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Na+ Concentration [mM]')
        ax2.legend()
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(figsize=figsize)
        ax3.plot(times, self.K_i_n_arr, label='Neuron K+ Intracellular')
        ax3.plot(times, self.K_i_g_arr, label='Glia K+ Intracellular')
        ax3.plot(times, self.K_e_arr, label='Extracellular K+')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('K+ Concentration [mM]')
        ax3.legend()
        fig3.tight_layout()

        fig4, ax4 = plt.subplots(figsize=figsize)
        ax4.plot(times, self.Cl_i_n_arr, label='Neuron Cl- Intracellular')
        ax4.plot(times, self.Cl_i_g_arr, label='Glia Cl- Intracellular')
        ax4.plot(times, self.Cl_e_arr, label='Extracellular Cl-')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Cl- Concentration [mM]')
        ax4.legend()
        fig4.tight_layout()

        fig5, ax5 = plt.subplots(figsize=figsize)
        ax5.plot(times, self.n_arr, label='n gating variable')
        ax5.plot(times, self.m_arr, label='m gating variable')
        ax5.plot(times, self.h_arr, label='h gating variable')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Gating Variables')
        ax5.legend()
        fig5.tight_layout()

        if self.plot_save:
            fig1.savefig('membrane_potential.png')
            fig2.savefig('na_concentration.png')
            fig3.savefig('k_concentration.png')
            fig4.savefig('cl_concentration.png')
            fig5.savefig('gating_variables.png')
        if self.plot_show:
            plt.show()


class TwoCompartmentMembraneODESystem(MembraneODESystem):
    """ Class to solve the ODE system for a two-compartment model (neuron + ECS). """

    def initialize_constants(self):
        """ Initialize constants and initial guesses for the ODE system. """
        # Initialize plotting arrays if enabled
        if self.plot:
            self.initialize_plotting()

        # Initialize constants and initial guesses
        # from the problem instance
        p = self.problem # For brevity
        self.R = p.R.value # Gas constant [J/(mol*K)]
        self.F = p.F.value # Faraday's constant [C/mol]
        self.T = p.T.value # Temperature [K]
        self.C_M = p.C_M.value # Membrane capacitance [F/m**2]
        self.g_Na_bar = p.g_Na_bar.value # Na max conductivity [S/m**2]
        self.g_K_bar = p.g_K_bar.value # K max conductivity [S/m**2]
        self.g_Na_leak = p.g_Na_leak.value # Na leak conductivity [S/m**2]
        self.g_K_leak = p.g_K_leak.value # K leak conductivity [S/m**2]
        self.g_Cl_leak = p.g_Cl_leak.value # Cl leak conductivity [S/m**2]
        self.phi_rest = p.phi_rest.value # Resting potential [V]
        self.phi_m_init = p.phi_m_init.value # Initial neuronal membrane potential [V]
        self.Na_i_init = p.Na_i_init.value # Initial neuronal intracellular Na+ concentration [mM]
        self.Na_e_init = p.Na_e_init.value # Initial extracellular Na+ concentration [mM]
        self.K_i_init = p.K_i_init.value # Initial neuronal intracellular K+ concentration [mM]
        self.K_e_init = p.K_e_init.value # Initial extracellular K+ concentration [mM]
        self.Cl_i_init = p.Cl_i_init.value # Initial neuronal intracellular Cl- concentration [mM]
        self.Cl_e_init = p.Cl_e_init.value # Initial extracellular Cl- concentration [mM]

        if self.stimulus:
            # Stimulus parameters
            g_syn_bar = p.g_syn_bar.value # Max synaptic conductance [S/m^2]
            T = p.T_stim.value # Stimulus period [s]
            a_syn = p.a_syn.value # Synaptic time constant [s]
            self.g_syn_fac = lambda t: g_syn_bar * np.exp((-np.mod(t+1e-10, T)) / a_syn)

    def initialize_initial_conditions(self, init_cond_array: list[float]):
        """ Initialize the initial conditions from an array.

            Parameters
            ----------
            init_cond_array : list[float]
                Array of initial conditions for the ODE system
        """
        (
            self.phi_m_init,
            self.Na_i_init,
            self.Na_e_init,
            self.K_i_init,
            self.K_e_init,
            self.Cl_i_init,
            self.Cl_e_init,
            self.n_init,
            self.m_init,
            self.h_init
        ) = init_cond_array
        
    def solve_ode_system(self) -> list[float]:

        p = self.problem # For brevity
        # Get constants
        R = self.R # Gas constant [J/(mol*K)]
        F = self.F # Faraday's constant [C/mol]
        T = self.T # Temperature [K]
        C_m = self.C_M # Membrane capacitance
        z_Na =  1 # Valence sodium
        z_K  =  1 # Valence potassium
        z_Cl = -1 # Valence chloride
        g_Na_stim_func = self.g_syn_fac if self.stimulus else 0.0
        g_Na_bar  = self.g_Na_bar                 # Na max conductivity (S/m**2)
        g_K_bar   = self.g_K_bar                  # K max conductivity (S/m**2)    
        g_Na_leak = self.g_Na_leak              # Na leak conductivity (S/m**2) (Constant)
        g_K_leak  = self.g_K_leak              # K leak conductivity (S/m**2)
        g_Cl_leak = self.g_Cl_leak                  # Cl leak conductivity (S/m**2) (Constant)
        phi_rest = self.phi_rest  # Resting potential [V]

        # Define initial condition guesses
        Na_i_0 = self.Na_i_init # [Mm]
        Na_e_0 = self.Na_e_init # [Mm]
        K_i_0 = self.K_i_init # [Mm]
        K_e_0 = self.K_e_init # [Mm]
        Cl_i_0 = self.Cl_i_init # [Mm]
        Cl_e_0 = self.Cl_e_init # [Mm]
        phi_m_0 = self.phi_m_init # [V]

        # ATP pump
        I_hat = 0.25 # Maximum pump strength [A/m^2]
        P_Na_i = 10          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
        P_K_e  = 1.5         # [K+]e  threshold for Na+/K+ pump (mol/m^3)

        # Cotransporters
        S_KCC2 = 0.0068
        S_NKCC1 = 0.00023

        # Hodgkin-Huxley parameters
        alpha_n = lambda V_m: 0.01e3 * (10.-V_m) / (np.exp((10. - V_m)/10.) - 1.)
        beta_n  = lambda V_m: 0.125e3 * np.exp(-V_m/80.)
        alpha_m = lambda V_m: 0.1e3 * (25. - V_m) / (np.exp((25. - V_m)/10.) - 1)
        beta_m  = lambda V_m: 4.e3 * np.exp(-V_m/18.)
        alpha_h = lambda V_m: 0.07e3 * np.exp(-V_m/20.)
        beta_h  = lambda V_m: 1.e3 / (np.exp((30. - V_m)/10.) + 1)

        # Set steady-state gating variables as initial guess
        V_m_gating = (phi_m_0 - phi_rest)*1e3 # Relative potential with unit correction
        n_0 = alpha_n(V_m_gating) / (alpha_n(V_m_gating) + beta_n(V_m_gating))
        m_0 = alpha_m(V_m_gating) / (alpha_m(V_m_gating) + beta_m(V_m_gating))
        h_0 = alpha_h(V_m_gating) / (alpha_h(V_m_gating) + beta_h(V_m_gating))

        # Nernst potential
        E = lambda z_k, c_ki, c_ke: R*T/(z_k*F) * np.log(c_ke/c_ki)

        # ATP current
        par_1 = lambda K_e: 1 + P_K_e / K_e
        par_2 = lambda Na_i: 1 + P_Na_i / Na_i
        I_ATP = lambda Na_i, K_e: \
                    I_hat / (par_1(K_e)**2 * par_2(Na_i)**3)

        # Cotransporter currents
        I_KCC2 = lambda K_i, K_e, Cl_i, Cl_e: \
                    S_KCC2 * np.log((K_i * Cl_i)/(K_e*Cl_e))
        I_NKCC1_n = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: \
                        S_NKCC1 * self.f_NKCC1(K_e, K_e_0) * np.log((Na_e * K_e * Cl_e**2)/(Na_i * K_i * Cl_i**2))

        # Volumes and surface areas in m^3 and m^2
        # Both neuronal and glial intracellular space
        vol_i_n = p.vol_i_n # [m^3]
        vol_e = p.vol_e # [m^3]
        area_g_n = p.area_g_n # [m^2]
        
        # Define right-hand side of ODE system
        def two_compartment_rhs(t: float, x: list[float]) -> list[float]:
            """ Right-hand side of ODE system for three-compartment system (neuron + ECS). 
            
                Parameters
                ----------
                t : float
                    Current time [s]
                x : list[float]
                    Current state vector
            """
            # Extract variables at previous timestep
            phi_m_n_ = x[0]; Na_i_n_ = x[1]; Na_e_ = x[2]; K_i_n_ = x[3]; K_e_ = x[4]; Cl_i_n_ = x[5]; Cl_e_ = x[6]
            n = x[7]; m = x[8]; h = x[9]

            # Neuronal mechanisms
            # Define potential used in gating variable expressions
            phi_m_gating = (phi_m_n_ - phi_rest)*1e3 # Relative potential with unit correction

            # Calculate Nernst potentials
            E_Na_n = E(z_Na, Na_i_n_, Na_e_)
            E_K_n  = E(z_K, K_i_n_, K_e_)
            E_Cl_n = E(z_Cl, Cl_i_n_, Cl_e_)

            g_Na_stim = g_Na_stim_func(t) if self.stimulus else 0.0

            # Calculate neuronal ionic currents
            I_Na_n = (
                    (g_Na_leak + g_Na_bar * m**3 * h + g_Na_stim) * (phi_m_n_ - E_Na_n)
                    + 3*I_ATP(Na_i_n_, K_e_)
                    - I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                )
            I_K_n = (
                    (g_K_leak + g_K_bar * n**4)* (phi_m_n_ - E_K_n)
                    - 2*I_ATP(Na_i_n_, K_e_)
                    - I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                    + I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                )
            I_Cl_n = (
                    g_Cl_leak * (phi_m_n_ - E_Cl_n)
                    + 2*I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                    - I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                )
            # Total neuronal ionic current [A/m^2]
            I_ion_n = I_Na_n + I_K_n + I_Cl_n

            # Define right-hand expressions
            rhs_phi_n = -1/C_m * I_ion_n
            rhs_Na_i_n = -I_Na_n/(z_Na*F) * area_g_n / vol_i_n
            rhs_Na_e =  I_Na_n/(z_Na*F) * area_g_n / vol_e
            rhs_K_i_n = -I_K_n/(z_K*F) * area_g_n / vol_i_n
            rhs_K_e =  I_K_n/(z_K*F) * area_g_n / vol_e
            rhs_Cl_i_n = -I_Cl_n/(z_Cl*F) * area_g_n / vol_i_n
            rhs_Cl_e =  I_Cl_n/(z_Cl*F) * area_g_n / vol_e
            rhs_n = alpha_n(phi_m_gating) * (1 - n) - beta_n(phi_m_gating) * n
            rhs_m = alpha_m(phi_m_gating) * (1 - m) - beta_m(phi_m_gating) * m
            rhs_h = alpha_h(phi_m_gating) * (1 - h) - beta_h(phi_m_gating) * h

            return [
                rhs_phi_n, rhs_Na_i_n, rhs_Na_e, rhs_K_i_n, rhs_K_e, rhs_Cl_i_n, rhs_Cl_e, # Neuronal variables
                rhs_n, rhs_m, rhs_h # Gating variables
                ]

        init = [
            phi_m_0, Na_i_0, Na_e_0, K_i_0, K_e_0, Cl_i_0, Cl_e_0,
            n_0, m_0, h_0,
                ]
        sol_ = init

        # Add initial condition to plotting arrays
        if self.plot: self.append_arrays(sol_)

        # Loop over timesteps and solve the ODE system
        for t, dt in zip(self.times, np.diff(self.times)):
        
            if t > 0:
                # Initial condition for ODE solver is
                # solution at previous timestep
                init = sol_

            # Integrate ODE system
            sol = solve_ivp(
                    lambda t, x: two_compartment_rhs(t, x),
                    [t, t+dt],
                    init,
                    method='Radau',
                    rtol=1e-6,
                    atol=1e-8
                )
            
            # Update previous solution
            sol_: list[float] = sol.y[:, -1] 

            if self.plot: self.append_arrays(sol_)
            
            if np.allclose(two_compartment_rhs(t, sol_), 0.0, rtol=1e-8, atol=1e-10):
                print("Steady state reached. Derivatives zero to within tolerance.")
                [print(f"Variable {j}: {sol_[j]:.18f}") for j in range(len(sol_))]
                break

            # Checks
            if np.isclose(t+dt, self.max_time):
                print("Max time exceeded without finding steady state. Exiting.")
                break

            if any(np.isnan(sol_)):
                print("NaN values in solution. Exiting.")
                break
        
        if self.plot:
            self.plot_results()

        return sol_
    
    def initialize_plotting(self):
        """ Initialize arrays for plotting results. """
        
        self.phi_m_n_arr = []
        self.Na_i_n_arr = []
        self.Na_e_arr = []
        self.K_i_n_arr = []
        self.K_e_arr = []
        self.Cl_i_n_arr = []
        self.Cl_e_arr = []
        self.n_arr = []
        self.m_arr = []
        self.h_arr = []

    def append_arrays(self, sol_: list[float]):
        """ Append current solution to plotting arrays.

            Parameters
            ----------
            sol_ : list[float]
                Current solution vector
        """
        self.phi_m_n_arr.append(sol_[0])
        self.Na_i_n_arr.append(sol_[1])
        self.Na_e_arr.append(sol_[2])
        self.K_i_n_arr.append(sol_[3])
        self.K_e_arr.append(sol_[4])
        self.Cl_i_n_arr.append(sol_[5])
        self.Cl_e_arr.append(sol_[6])
        self.n_arr.append(sol_[7])
        self.m_arr.append(sol_[8])
        self.h_arr.append(sol_[9])

    def plot_results(self):
        """ Plot the results of the ODE system solution. 
        
            Parameters
            ----------
            show : bool, optional
                Whether to display the plots, by default False
            save : bool, optional
                Whether to save the plots as PNG files, by default False
        """
        times = np.arange(len(self.phi_m_n_arr)) * 1e-3 * self.timestep # Milliseconds
        figsize = (10, 6)

        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(times, self.phi_m_n_arr, label='Neuron Membrane Potential')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Membrane Potential [V]')
        ax1.legend()
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.plot(times, self.Na_i_n_arr, label='Neuron Na+ Intracellular')
        ax2.plot(times, self.Na_e_arr, label='Extracellular Na+')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Na+ Concentration [mM]')
        ax2.legend()
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(figsize=figsize)
        ax3.plot(times, self.K_i_n_arr, label='Neuron K+ Intracellular')
        ax3.plot(times, self.K_e_arr, label='Extracellular K+')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('K+ Concentration [mM]')
        ax3.legend()
        fig3.tight_layout()

        fig4, ax4 = plt.subplots(figsize=figsize)
        ax4.plot(times, self.Cl_i_n_arr, label='Neuron Cl- Intracellular')
        ax4.plot(times, self.Cl_e_arr, label='Extracellular Cl-')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Cl- Concentration [mM]')
        ax4.legend()
        fig4.tight_layout()

        fig5, ax5 = plt.subplots(figsize=figsize)
        ax5.plot(times, self.n_arr, label='n gating variable')
        ax5.plot(times, self.m_arr, label='m gating variable')
        ax5.plot(times, self.h_arr, label='h gating variable')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Gating Variables')
        ax5.legend()
        fig5.tight_layout()

        if self.plot_save:
            fig1.savefig('membrane_potential.png')
            fig2.savefig('na_concentration.png')
            fig3.savefig('k_concentration.png')
            fig4.savefig('cl_concentration.png')
            fig5.savefig('gating_variables.png')
        if self.plot_show:
            plt.show()

if __name__ == "__main__":
    from sys import argv
    from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI

    # Create a problem instance
    problem = ProblemKNPEMI(config_file=str(argv[1]) if len(argv) > 1 else "../KNPEMI/configs/5m/100c.yaml")
    problem.calculate_compartment_volumes_and_surface_areas()

    # Create a three-compartment ODE system solver
    ode_system = ThreeCompartmentMembraneODESystem(problem, timestep=1e-3, max_time=100.0, plot_show=True, plot_save=False)

    # Solve the ODE system
    solution = ode_system.solve_ode_system()