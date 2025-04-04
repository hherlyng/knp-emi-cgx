import ufl
import time

import numpy as np
import dolfinx as dfx

from abc import ABC, abstractmethod
from mpi4py import MPI
from petsc4py import PETSc


# Stimulus
def g_syn(g_syn_bar, a_syn, t: float) -> float:
    g = g_syn_bar * np.exp(-np.mod(t, 0.01) / a_syn)
    # g = g_syn_bar
    # g = lambda x: g_syn_bar*np.exp(-np.mod(t,0.01)/a_syn)*(x[2] < 5e-4)
    # g = lambda x: g_syn_bar*(x[0] < 0.3) * (x[1] < 0.3)

    return g


# Kir-function used in ionic pump
def f_Kir(K_e_init, K_e, EK_init, Dphi, phi_m):
    A = 1 + ufl.exp(18.4 / 42.4)
    B = 1 + ufl.exp(-(0.1186 + EK_init) / 0.0441)
    C = 1 + ufl.exp((Dphi + 0.0185) / 0.0425)
    D = 1 + ufl.exp(-(0.1186 + phi_m) / 0.0441)

    f = ufl.sqrt(K_e / K_e_init) * A * B / (C * D)

    return f


#################################


class IonicModel(ABC):
    # constructor
    def __init__(self, KNPEMIx_problem, tags=None):
        self.problem = KNPEMIx_problem
        self.tags = tags

        # if tags are not specified we use all the intra tags
        if self.tags == None:
            self.tags = self.problem.gamma_tags

        # trasform int in tuple if needed
        if isinstance(self.tags, int):
            self.tags = (self.tags,)

    @abstractmethod
    def _init(self):
        # Abstract method that must be implemented by concrete subclasses.
        # Init ion-independent quantities
        pass

    @abstractmethod
    def _eval(self, ion_idx):
        # Abstract method that must be implemented by concrete subclasses.
        pass


# I_ch = phi_M
class Null_model(IonicModel):
    def _init(self):
        pass

    def __str__(self):
        return f"Zero"

    def _eval(self, ion_idx):
        return 0


# I_ch = phi_M
class Passive_model(IonicModel):
    def _init(self):
        pass

    def __str__(self):
        return f"Passive"

    def _eval(self, ion_idx):
        I_ch = self.problem.phi_M_prev
        return I_ch


# I_ch = g*(phi_M - E) + stimuls
class Passive_Nerst_model(IonicModel):
    def __init__(self, KNPEMIx_problem, tags=None, stimuls=True):
        super().__init__(KNPEMIx_problem, tags)

        self.stimuls = stimuls

    def __str__(self):
        return f"Passive"

    def _init(self):
        pass

    def _eval(self, ion_idx):
        # aliases
        p = self.problem
        ion = p.ion_list[ion_idx]
        phi_M = p.phi_M_prev

        # leak currents
        ion["g_k"] = ion["g_leak"]

        # stimulus
        if ion["name"] == "Na" and self.stimuls:
            ion["g_k"] += g_syn(p.g_syn_bar, p.a_syn, float(p.t))

        I_ch = ion["g_k"] * (phi_M - ion["E"])

        return I_ch


# Ionic K pump
class Passive_K_pump_model(IonicModel):
    # -k_dec * ([K]e − [K]e_0) both for K and Na
    use_decay_currents = False

    def __str__(self):
        if self.use_decay_currents:
            return f"Passive with K pump and decay currents"
        else:
            return f"Passive with K pump"

    def _init(self):
        # aliases
        p = self.problem

        ui_p = p.u_p[0]
        ue_p = p.u_p[1]
        P_Nai = p.P_Nai
        P_Ke = p.P_Ke

        self.pump_coeff = (
            ui_p.sub(0) ** 1.5
            / (ui_p.sub(0) ** 1.5 + P_Nai**1.5)
            * (ue_p.sub(1) / (ue_p.sub(1) + P_Ke))
        )

    def _eval(self, ion_idx):
        # aliases
        p = self.problem

        phi_M = p.phi_M_prev
        ue_p = p.u_p[1]
        F = p.F
        ion = p.ion_list[ion_idx]
        z = ion["z"]
        K_e_init = p.K_e_init

        # leak currents
        ion["g_k"] = ion["g_leak"]

        # f kir coeff
        if ion["name"] == "K":
            EK_init = (p.psi / z) * ufl.ln(K_e_init / p.K_i_init)
            Dphi = phi_M - ion["E"]
            f_kir = f_Kir(K_e_init, ue_p.sub(ion_idx), EK_init, Dphi, phi_M)

        else:
            f_kir = 1

        I_ch = (
            f_kir * ion["g_k"] * (phi_M - ion["E"])
            + F * z * ion["rho_p"] * self.pump_coeff
        )

        if self.use_decay_currents:
            if ion["name"] == "K" or ion["name"] == "Na":
                I_ch -= F * z * p.k_dec * (ue_p.sub(1) - K_e_init)

        return I_ch


# Hodgkin–Huxley + stimuls
class HH_model(IonicModel):
    def __init__(
        self,
        KNPEMIx_problem,
        tags=None,
        stimuls: bool = True,
        use_Rush_Lar: bool = True,
        time_steps_ODE: int = 25,
    ):
        super().__init__(KNPEMIx_problem, tags)

        self.stimuls = stimuls
        self.use_Rush_Lar = use_Rush_Lar
        self.time_steps_ODE = time_steps_ODE

    def __str__(self):
        return f"Hodgkin-Huxley"

    def _init(self):
        # alias
        p = self.problem

        # update gating variables
        if np.isclose(float(p.t), 0):
            G, _ = p.V.sub(p.N_ions).collapse()  # Gating function finite element space
            p.n = dfx.fem.Function(G)
            p.m = dfx.fem.Function(G)
            p.h = dfx.fem.Function(G)

            p.n.x.array[:] = p.n_init_val
            p.m.x.array[:] = p.m_init_val
            p.h.x.array[:] = p.h_init_val

        else:
            self.update_gating_variables()

    def _eval(self, ion_idx: int):
        """Evaluate and return the passive channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        I_ch : float
            The value of the passive channel current.
        """
        # aliases
        p = self.problem
        ion = p.ion_list[ion_idx]
        phi_M = p.phi_M_prev

        # leak currents
        ion["g_k"] = ion["g_leak"]

        # stimulus and gating
        if ion["name"] == "Na":
            ion["g_k"] += p.g_Na_bar * p.m**3 * p.h

        elif ion["name"] == "K":
            ion["g_k"] += p.g_K_bar * p.n**4

        I_ch = ion["g_k"] * (phi_M - ion["E"])

        return I_ch

    def _add_stimulus(self, ion_idx: int):
        """Evaluate and return the stimulus part of the channel current for ion number 'ion_idx'.

        Parameters
        ----------
        ion_idx : int
            Ion index.

        Returns
        -------
        type float
            The stimulus part of the channel current.
        """

        # aliases
        p = self.problem
        ion = p.ion_list[ion_idx]
        phi_M = p.phi_M_prev

        assert ion["name"] == "Na", print(
            "Only Na can have a stimulus current in the Hodgkin-Huxley model."
        )

        return g_syn(p.g_syn_bar, p.a_syn, float(p.t.value)) * (phi_M - ion["E"])

    def update_gating_variables(self):
        tic = time.perf_counter()

        # aliases
        n = self.problem.n
        m = self.problem.m
        h = self.problem.h
        phi_M_prev = self.problem.phi_M_prev
        dt_ode = float(self.problem.dt.value) / self.time_steps_ODE

        # Set membrane potential
        with phi_M_prev.x.petsc_vec.localForm() as loc_phi_M_prev:
            V_M = 1000 * (
                loc_phi_M_prev[:] - self.problem.V_rest
            )  # convert phi_M to mV

        alpha_n = 0.01e3 * (10.0 - V_M) / (np.exp((10.0 - V_M) / 10.0) - 1.0)
        beta_n = 0.125e3 * np.exp(-V_M / 80.0)
        alpha_m = 0.1e3 * (25.0 - V_M) / (np.exp((25.0 - V_M) / 10.0) - 1)
        beta_m = 4.0e3 * np.exp(-V_M / 18.0)
        alpha_h = 0.07e3 * np.exp(-V_M / 20.0)
        beta_h = 1.0e3 / (np.exp((30.0 - V_M) / 10.0) + 1)

        if self.use_Rush_Lar:
            tau_y_n = 1.0 / (alpha_n + beta_n)
            tau_y_m = 1.0 / (alpha_m + beta_m)
            tau_y_h = 1.0 / (alpha_h + beta_h)

            y_inf_n = alpha_n * tau_y_n
            y_inf_m = alpha_m * tau_y_m
            y_inf_h = alpha_h * tau_y_h

            y_exp_n = np.exp(-dt_ode / tau_y_n)
            y_exp_m = np.exp(-dt_ode / tau_y_m)
            y_exp_h = np.exp(-dt_ode / tau_y_h)

        else:
            alpha_n *= dt_ode
            beta_n *= dt_ode
            alpha_m *= dt_ode
            beta_m *= dt_ode
            alpha_h *= dt_ode
            beta_h *= dt_ode

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
        ODE_step_time = self.problem.mesh.comm.allreduce(toc - tic, op=MPI.MAX)
        PETSc.Sys.Print(f"ODE step in {ODE_step_time:0.4f} seconds")
