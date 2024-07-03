import time

import numpy   as np
import dolfinx as dfx

from abc      import ABC, abstractmethod
from mpi4py   import MPI
from petsc4py import PETSc

# zero stimulus (default)
def g_syn_none(t: float) -> float:		
	return 0.0

# Stimulus
def g_syn(t: float) -> float:
    a_syn     = 0.002           
    g_syn_bar = 40 
    g = g_syn_bar * np.exp(-np.mod(t, 0.01)/a_syn)
    # g = g_syn_bar			
    # g = lambda x: g_syn_bar*np.exp(-np.mod(t,0.01)/a_syn)*(x[2] < 5e-4)
    # g = lambda x: g_syn_bar*(x[0] < 0.3) * (x[1] < 0.3)
    
    return g

#################################

class IonicModel(ABC):

	# constructor
	def __init__(self, EMIx_problem, tags: int | tuple | dict=None):

		self.problem = EMIx_problem	
		self.tags = tags

		# if tags are not specified we use all the intra tags
		if self.tags == None:
			self.tags = self.problem.gamma_tags

		# trasform int in tuple if needed
		if isinstance(self.tags, int): self.tags = (self.tags,)
			
	@abstractmethod
	def _eval(self, ion_idx):
		# Abstract method that must be implemented by concrete subclasses.
		pass


# I_ch = phi_M
class Passive_model(IonicModel):

	def _init(self):		
		pass

	def __str__(self):
		return f'Passive'
		
	def _eval(self, ion_idx):	
		I_ch = self.problem.phi_M_prev	
		return I_ch

# Hodgkin–Huxley + stimulus
class HH_model(IonicModel):

    # HH params
    # initial gating variables
    n_init_val = 0.27622914792 # gating variable n
    m_init_val = 0.03791834627 # gating variable m
    h_init_val = 0.68848921811 # gating variable h

    # conductivities
    g_Na_bar  = 1200       # Na max conductivity (S/m**2)
    g_K_bar   = 360        # K max conductivity (S/m**2)    
    g_Na_leak = 2.0*0.5    # Na leak conductivity (S/m**2)
    g_K_leak  = 8.0*0.5    # K leak conductivity (S/m**2)
    g_Cl_leak = 0.0        # Cl leak conductivity (S/m**2)		
    V_rest    = -0.065     # resting membrane potential
    E_Na      = 54.8e-3    # reversal potential Na (V)
    E_K       = -88.98e-3  # reversal potential K (V)
    E_Cl      = 0  		   # reversal potential 0 (V)

    # numerics
    use_Rush_Lar   = True
    time_steps_ODE = 25

    # save gating in PNG	
    save_png_file = True		

    def __init__(self, EMIx_problem, tags: int | tuple | dict=None, stim_fun=g_syn_none):

        super().__init__(EMIx_problem, tags)

        self.g_Na_stim = stim_fun

    def __str__(self):
        return f'Hodgkin-Huxley'

    def _eval(self):
        # Alias
        p = self.problem

        if float(p.t.value==0):
            # Initial timestep

            p.n = dfx.fem.Function(p.V)
            p.m = dfx.fem.Function(p.V)
            p.h = dfx.fem.Function(p.V)

            p.n.x.array[:] = self.n_init_val
            p.m.x.array[:] = self.m_init_val
            p.h.x.array[:] = self.h_init_val

            # output
            #if self.save_png_file: self.init_png()

        else:
            self.update_gating_variables()	

            # output
            #if self.save_png_file: self.save_png()					

        # conductivities
        g_Na = self.g_Na_leak + self.g_Na_bar*p.m**3*p.h
        g_K  = self.g_K_leak  + self.g_K_bar*p.n**4				
        g_Cl = self.g_Cl_leak

        # stimulus
        g_Na += self.g_Na_stim(float(p.t.value)) 

        # ionic currents
        I_ch_Na = g_Na * (p.phi_M_prev - self.E_Na)
        I_ch_K  = g_K  * (p.phi_M_prev - self.E_K)
        I_ch_Cl = g_Cl * (p.phi_M_prev - self.E_Cl)		

        # total current
        I_ch = I_ch_Na + I_ch_K + I_ch_Cl

        return I_ch

    def update_gating_variables(self):		

        tic = time.perf_counter()

        # aliases	
        n = self.problem.n
        m = self.problem.m
        h = self.problem.h
        phi_M_prev = self.problem.phi_M_prev
        dt_ode     = float(self.problem.dt.value) / self.time_steps_ODE
        
        # Set membrane potential
        with phi_M_prev.vector.localForm() as loc_phi_M_prev:
            V_M = 1000*(loc_phi_M_prev[:] - self.V_rest) # convert phi_M to mV	
        
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
                with n.vector.localForm() as loc_n, m.vector.localForm() as loc_m, h.vector.localForm() as loc_h:

                    loc_n[:] = y_inf_n + (loc_n[:] - y_inf_n) * y_exp_n
                    loc_m[:] = y_inf_m + (loc_m[:] - y_inf_m) * y_exp_m
                    loc_h[:] = y_inf_h + (loc_h[:] - y_inf_h) * y_exp_h

            else:
                # Get vector entries local to process + ghosts
                with n.vector.localForm() as loc_n, m.vector.localForm() as loc_m, h.vector.localForm() as loc_h:

                    loc_n[:] += alpha_n * (1 - loc_n[:]) - beta_n * loc_n[:]
                    loc_m[:] += alpha_m * (1 - loc_m[:]) - beta_m * loc_m[:]
                    loc_h[:] += alpha_h * (1 - loc_h[:]) - beta_h * loc_h[:]	

        toc = time.perf_counter()
        ODE_step_time = MPI.COMM_WORLD.allreduce(toc-tic, op=MPI.MAX)
        PETSc.Sys.Print(f"ODE step in {ODE_step_time:0.4f} seconds")   	