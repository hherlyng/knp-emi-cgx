import ufl
import time
import numpy.typing
import collections.abc

import numpy   as np
import sympy   as sp
import dolfinx as dfx

from abc                      import ABC, abstractmethod
from scipy                    import sparse
from mpi4py                   import MPI
from petsc4py                 import PETSc
from sympy.utilities.lambdify import lambdify

def norm_2(vec):
    return ufl.sqrt(ufl.dot(vec, vec))

def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)
            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])

def flatten_list(input_list):
    return [item for sublist in input_list for item in (sublist if isinstance(sublist, tuple) else [sublist])]

class MixedDimensionalProblem(ABC):

    ghost_mode = dfx.mesh.GhostMode.shared_facet

    def __init__(self, input_file, tags, dt):
        
        tic = time.perf_counter()

        self.comm = MPI.COMM_WORLD

        if MPI.COMM_WORLD.rank==0: print("Reading input data: ", input_file)

        # Options for the fenicsx form compiler optimization
        cache_dir       = '.cache'
        compile_options = ["-Ofast", "-march=native"]
        self.jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                                "cache_dir"               : cache_dir,
                                "cffi_libraries"          : ["m"]}

        # assign input argument
        self.input_file = input_file

        # parse tags
        self.parse_tags(tags)

        # in case some problem dependent init is needed
        self.init()

        # setup FEM
        self.setup_domain()
        self.setup_spaces()
        self.setup_boundary_conditions()

        # init time step
        self.t  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.dt = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(dt))

        # init empty ionic model
        self.ionic_models = []

        if MPI.COMM_WORLD.rank==0: print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds.\n")

    def parse_tags(self, tags):

        allowed_tags = {'intra', 'extra', 'membrane', 'boundary'}

        tags_set = set(tags.keys())

        if MPI.COMM_WORLD.rank==0:

            # checks
            if not tags_set.issubset(allowed_tags):
                raise ValueError(f'Mismatch in tags.\nAllowed tags: {allowed_tags}\nInput tags: {tags_set}')

            # print info
            if isinstance(tags['intra'], collections.abc.Sequence):
                print(f"# Cell tags = {len(tags['intra'])}.")           
            else:           
                print("Single cell tag.")    

        if 'intra' in tags_set:
            self.intra_tags = tags['intra']
        else:
            if MPI.COMM_WORLD.rank==0: raise ValueError('Intra tag has to be provided.')
        
        if 'extra' in tags_set:
            self.extra_tag = tags['extra']
        else:
            if MPI.COMM_WORLD.rank==0: print('Setting default: extra tag = 1.')
            self.extra_tag = 1
        
        if 'membrane' in tags_set:
            self.gamma_tags = tags['membrane']
        else:
            if MPI.COMM_WORLD.rank==0: print('Setting default: membrane tag = intra tag.')
            self.gamma_tags = self.intra_tags
        
        if 'boundary' in tags_set:
            self.boundary_tag = tags['boundary']
        else:
            if MPI.COMM_WORLD.rank==0: print('Setting default: boundary tag = 1.')

        # Transform ints to tuples if needed
        if isinstance(self.intra_tags, int): self.intra_tags = (self.intra_tags,)
        if isinstance(self.gamma_tags, int): self.gamma_tags = (self.gamma_tags,)

    def init_ionic_model(self, ionic_models):

        self.ionic_models = ionic_models

        # init list
        ionic_tags = []
    
        # check all intracellular space tags are present in some ionic model
        for model in ionic_models:
            ionic_tags.append(model.tags)
        
        ionic_tags = sorted(flatten_list(ionic_tags))
        gamma_tags = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags:
            raise RuntimeError('Mismatch between membrane tags and ionic models tags.' \
                + f'\nIonic models tags: {ionic_tags}\nMembrane tags: {gamma_tags}')
        
        if MPI.COMM_WORLD.rank==0:
            print('# Membrane tags = ', len(gamma_tags))
            print('# Ionic models  = ', len(ionic_models), '\n')

    def setup_domain(self):

        if MPI.COMM_WORLD.rank==0: print("Reading mesh from XDMF file...")

        # Rename file for readability
        mesh_file = self.input_file


        if not self.MMS_test:
            # Load mesh files with meshtags
            
            with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                # Read mesh and cell tags
                self.mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode, name="mesh")
                self.subdomains = xdmf.read_meshtags(self.mesh, name="ct")
                self.subdomains.name = "ct"

                # Create facet-to-cell connectivity
                self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)

                # Read facet tags
                self.boundaries = xdmf.read_meshtags(self.mesh, name="ft")
                self.boundaries.name = "ft"      
            
            # Scale mesh
            self.mesh.geometry.x[:] *= self.m_conversion_factor
        
        else:
            self.dim = 2
            if self.dim == 2:
                self.mesh = dfx.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx = self.N_mesh, ny = self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_square(self.mesh)
                self.boundaries = mark_boundaries_square_MMS(self.mesh)
            
            elif self.dim == 3:
                self.mesh = dfx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx = self.N_mesh, ny = self.N_mesh, nz = self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_cube(self.mesh)
                self.boundaries = mark_boundaries_cube_MMS(self.mesh)

        # Integral measures for the domain
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains) # Volume integral measure
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.boundaries) # Facet integral measure
    
    @abstractmethod
    def init(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass
    
    @abstractmethod
    def setup_spaces(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass

    @abstractmethod
    def setup_boundary_conditions(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass

## MMS Functions
class SetupMMS:
    """ class for calculating source terms of the KNP-EMI system for given exact
    solutions """
    def __init__(self, mesh):
        self.mesh = mesh
        self.dim  = mesh.topology.dim
        # define symbolic variables
        if self.dim == 2:
            self.x, self.y, self.t = sp.symbols('x[0] x[1] t')
        elif self.dim == 3:
            self.x, self.y, self.z, self.t = sp.symbols('x[0] x[1] x[2] t')

    def get_exact_solution(self):
        """ define manufactured (exact) solutions """
        x = self.x; y = self.y; t = self.t

        if self.dim == 2:
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

        elif self.dim == 3:
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

        exact_solutions = {'Na_i_e':Na_i_e, 'K_i_e':K_i_e, 'Cl_i_e':Cl_i_e,\
                           'Na_e_e':Na_e_e, 'K_e_e':K_e_e, 'Cl_e_e':Cl_e_e,\
                           'phi_i_e':phi_i_e, 'phi_e_e':phi_e_e}

        return exact_solutions

    def get_MMS_terms_KNPEMI_2D(self, time):
        """ get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            KNP-EMI problem """
        # Variables
        x = self.x; y = self.y; t = self.t

        # get manufactured solution
        exact_solutions = self.get_exact_solution()
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
        phi_M_e = phi_i_e - phi_e_e

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
        I_ch_Na = phi_M_e                 # Na
        I_ch_K  = phi_M_e                 # K
        I_ch_Cl = phi_M_e                 # Cl
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

        # equation for phi_M: f = C_M*d(phi_M)/dt - (I_M - I_ch) where we have
        # chosen I_M = F sum_k z^k J_i^k n_i = total_flux_i
        fJM = [sp.diff(phi_M_e, t) + I_ch - foo for foo in JMe_i]
        # coupling condition for I_M: (total_flux_i*n_i) = (- total_flux_e*n_e) + f
        # giving f = total_flux_i*n_i + total_flux_e*n_e
        fgM = [i - e for i, e in zip(JMe_i, JMe_e)]


        # Class for converting symbolic SymPy functions to DOLFINx expressions
        class SymPyToDOLFINxExpr():
            def __init__(self, sp_func, time, dim):
                self.time = time
                self.dim = dim
                self.f = lambdify([x, y, t], sp_func)
                
            def __call__(self, x):
                return self.f(x[0], x[1], self.time)

        ##### Convert to expressions with exact solutions #####
        # Create P1 space for all functions
        V = dfx.fem.FunctionSpace(self.mesh, ("Lagrange", 1))

        # Ion concentrations and electric potentials
        var_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_i_e, phi_e_e, phi_M_e]
        var_exprs = [SymPyToDOLFINxExpr(var_func, time, self.dim) for var_func in var_sym_funcs]
        var_dfx_funcs = [dfx.fem.Function(V) for _ in var_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(var_dfx_funcs, var_exprs)]
        Nai_e, Nae_e, Ki_e, Ke_e, Cli_e, Cle_e, phii_e, phie_e, phiM_e = [var_dfx_funcs[i] for i in range(len(var_dfx_funcs))]

        # Membrane flux
        JM_e_exprs = [SymPyToDOLFINxExpr(JMe_func, time, self.dim) for JMe_func in JMe_i]
        JM_e = [dfx.fem.Function(V) for _ in range(len(JM_e_exprs))]
        [JM_e[i].interpolate(JM_e_exprs[i]) for i in range(len(JM_e_exprs))]

        # source terms
        source_sym_funcs = [f_Na_i, f_Na_e, f_K_i, f_K_e, f_Cl_i, f_Cl_e, f_phi_i, f_phi_e]
        source_exprs = [SymPyToDOLFINxExpr(source_func, time, self.dim) for source_func in source_sym_funcs]
        source_dfx_funcs = [dfx.fem.Function(V) for _ in source_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(source_dfx_funcs, source_exprs)]
        f_Nai, f_Nae, f_Ki, f_Ke, f_Cli, f_Cle, f_phii, f_phie = [source_dfx_funcs[i] for i in range(len(source_dfx_funcs))]

        # source term membrane flux
        f_JM_exprs = [SymPyToDOLFINxExpr(fJM_func, time, self.dim) for fJM_func in fJM]
        f_JM = [dfx.fem.Function(V) for _ in f_JM_exprs]
        [f_JM[i].interpolate(f_JM_exprs[i]) for i in range(len(f_JM_exprs))]

        # source term continuity coupling condition on gamma
        f_gM_exprs = [SymPyToDOLFINxExpr(fgM_func, time, self.dim) for fgM_func in fgM]
        f_gM = [dfx.fem.Function(V) for _ in f_gM_exprs]
        [f_gM[i].interpolate(f_gM_exprs[i]) for i in range(len(f_gM_exprs))]

        # initial conditions concentrations
        init_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_M_e]
        init_exprs = [SymPyToDOLFINxExpr(init_func, time, self.dim) for init_func in init_sym_funcs]
        init_dfx_funcs = [dfx.fem.Function(V) for _ in init_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(init_dfx_funcs, init_exprs)]
        init_Nai, init_Nae, init_Ki, init_Ke, init_Cli, init_Cle, init_phiM = [init_dfx_funcs[i] for i in range(len(init_dfx_funcs))]

        # exterior boundary terms
        import ufl
        P1_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), degree=1)
        V_vec = dfx.fem.FunctionSpace(self.mesh, element=P1_vec)
        ext_sym_funcs = [J_Na_e, J_K_e, J_Cl_e]
        ext_exprs = [SymPyToDOLFINxExpr(ext_func, time, self.dim) for ext_func in ext_sym_funcs]
        ext_dfx_funcs = [dfx.fem.Function(V_vec) for _ in ext_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ext_dfx_funcs, ext_exprs)]
        J_Nae, J_Ke, J_Cle = [ext_dfx_funcs[i] for i in range(len(ext_dfx_funcs))]

        # ion channel currents
        ion_ch_sym_funcs = [I_ch_Na, I_ch_K, I_ch_Cl]
        ion_ch_exprs = [SymPyToDOLFINxExpr(ion_ch_func, time, self.dim) for ion_ch_func in ion_ch_sym_funcs]
        ion_ch_funcs = [dfx.fem.Function(V) for _ in ion_ch_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ion_ch_funcs, ion_ch_exprs)]
        I_ch_Na, I_ch_K, I_ch_Cl = [ion_ch_funcs[i] for i in range(len(ion_ch_funcs))]

        # Gather expressions
        # exact solutions
        exact_sols = {'Na_i_e':Nai_e, 'K_i_e':Ki_e, 'Cl_i_e':Cli_e,
                      'Na_e_e':Nae_e, 'K_e_e':Ke_e, 'Cl_e_e':Cle_e,
                      'phi_i_e':phii_e, 'phi_e_e':phie_e, 'phi_M_e':phiM_e,
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
                      'phi_M':init_phiM}
        # boundary terms
        bndry_terms = {'J_Na_e':J_Nae, 'J_K_e':J_Ke, 'J_Cl_e':J_Cle}

        return src_terms, exact_sols, init_conds, bndry_terms

    def get_MMS_terms_KNPEMI_3D(self, time):
        """ get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            KNP-EMI problem """
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
        phi_M_e = phi_i_e - phi_e_e

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
        I_ch_Na = phi_M_e                 # Na
        I_ch_K  = phi_M_e                 # K
        I_ch_Cl = phi_M_e                 # Cl
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

        # equation for phi_M: f = C_M*d(phi_M)/dt - (I_M - I_ch) where we have
        # chosen I_M = F sum_k z^k J_i^k n_i = total_flux_i
        fJM = [sp.diff(phi_M_e, t) + I_ch - foo for foo in JMe_i]
        # coupling condition for I_M: (total_flux_i*n_i) = (- total_flux_e*n_e) + f
        # giving f = total_flux_i*n_i + total_flux_e*n_e
        fgM = [i - e for i, e in zip(JMe_i, JMe_e)]


        # Class for converting symbolic SymPy functions to DOLFINx expressions
        class SymPyToDOLFINxExpr():
            def __init__(self, sp_func, time, dim):
                self.time = time
                self.dim = dim
                self.f = lambdify([x, y, z, t], sp_func)
                
            def __call__(self, x):
                return self.f(x[0], x[1], x[2], self.time)

        ##### Convert to expressions with exact solutions #####
        # Create P1 space for all functions
        V = dfx.fem.FunctionSpace(self.mesh, ("Lagrange", 1))

        # Ion concentrations and electric potentials
        var_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_i_e, phi_e_e, phi_M_e]
        var_exprs = [SymPyToDOLFINxExpr(var_func, time, self.dim) for var_func in var_sym_funcs]
        var_dfx_funcs = [dfx.fem.Function(V) for _ in var_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(var_dfx_funcs, var_exprs)]
        Nai_e, Nae_e, Ki_e, Ke_e, Cli_e, Cle_e, phii_e, phie_e, phiM_e = [var_dfx_funcs[i] for i in range(len(var_dfx_funcs))]

        # Membrane flux
        JM_e_exprs = [SymPyToDOLFINxExpr(JMe_func, time, self.dim) for JMe_func in JMe_i]
        JM_e = [dfx.fem.Function(V) for _ in range(len(JM_e_exprs))]
        [JM_e[i].interpolate(JM_e_exprs[i]) for i in range(len(JM_e_exprs))]

        # source terms
        source_sym_funcs = [f_Na_i, f_Na_e, f_K_i, f_K_e, f_Cl_i, f_Cl_e, f_phi_i, f_phi_e]
        source_exprs = [SymPyToDOLFINxExpr(source_func, time, self.dim) for source_func in source_sym_funcs]
        source_dfx_funcs = [dfx.fem.Function(V) for _ in source_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(source_dfx_funcs, source_exprs)]
        f_Nai, f_Nae, f_Ki, f_Ke, f_Cli, f_Cle, f_phii, f_phie = [source_dfx_funcs[i] for i in range(len(source_dfx_funcs))]

        # source term membrane flux
        f_JM_exprs = [SymPyToDOLFINxExpr(fJM_func, time, self.dim) for fJM_func in fJM]
        f_JM = [dfx.fem.Function(V) for _ in f_JM_exprs]
        [f_JM[i].interpolate(f_JM_exprs[i]) for i in range(len(f_JM_exprs))]

        # source term continuity coupling condition on gamma
        f_gM_exprs = [SymPyToDOLFINxExpr(fgM_func, time, self.dim) for fgM_func in fgM]
        f_gM = [dfx.fem.Function(V) for _ in f_gM_exprs]
        [f_gM[i].interpolate(f_gM_exprs[i]) for i in range(len(f_gM_exprs))]

        # initial conditions concentrations
        init_sym_funcs = [Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_M_e]
        init_exprs = [SymPyToDOLFINxExpr(init_func, time, self.dim) for init_func in init_sym_funcs]
        init_dfx_funcs = [dfx.fem.Function(V) for _ in init_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(init_dfx_funcs, init_exprs)]
        init_Nai, init_Nae, init_Ki, init_Ke, init_Cli, init_Cle, init_phiM = [init_dfx_funcs[i] for i in range(len(init_dfx_funcs))]

        # exterior boundary terms
        import ufl
        P1_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), degree=1)
        V_vec = dfx.fem.FunctionSpace(self.mesh, element=P1_vec)
        ext_sym_funcs = [J_Na_e, J_K_e, J_Cl_e]
        ext_exprs = [SymPyToDOLFINxExpr(ext_func, time, self.dim) for ext_func in ext_sym_funcs]
        ext_dfx_funcs = [dfx.fem.Function(V_vec) for _ in ext_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ext_dfx_funcs, ext_exprs)]
        J_Nae, J_Ke, J_Cle = [ext_dfx_funcs[i] for i in range(len(ext_dfx_funcs))]

        # ion channel currents
        ion_ch_sym_funcs = [I_ch_Na, I_ch_K, I_ch_Cl]
        ion_ch_exprs = [SymPyToDOLFINxExpr(ion_ch_func, time, self.dim) for ion_ch_func in ion_ch_sym_funcs]
        ion_ch_funcs = [dfx.fem.Function(V) for _ in ion_ch_sym_funcs]
        [func.interpolate(expr) for func, expr in zip(ion_ch_funcs, ion_ch_exprs)]
        I_ch_Na, I_ch_K, I_ch_Cl = [ion_ch_funcs[i] for i in range(len(ion_ch_funcs))]

        # Gather expressions
        # exact solutions
        exact_sols = {'Na_i_e':Nai_e, 'K_i_e':Ki_e, 'Cl_i_e':Cli_e,
                      'Na_e_e':Nae_e, 'K_e_e':Ke_e, 'Cl_e_e':Cle_e,
                      'phi_i_e':phii_e, 'phi_e_e':phie_e, 'phi_M_e':phiM_e,
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
                      'phi_M':init_phiM}
        # boundary terms
        bndry_terms = {'J_Na_e':J_Nae, 'J_K_e':J_Ke, 'J_Cl_e':J_Cle}

        return src_terms, exact_sols, init_conds, bndry_terms



## Meshing utilities
def mark_subdomains_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking subdomains of a unit square mesh with an interior square defined on [0.25, 0.75]^2.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y) = [0.25, 0.75]^2
        - tag value 2 : outer square, (x, y) = [0, 1]^2 \ [0.25, 0.75]^2
    
    """ 
    def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        """ Locator function for the inner square. """

        bool1 = np.logical_and(x[0] <= 0.75, x[0] >= 0.25) # True if inside inner box in x range
        bool2 = np.logical_and(x[1] <= 0.75, x[1] >= 0.25) # True if inside inner box in y range
        
        return np.logical_and(bool1, bool2)

    # Tag values
    INTRA = 1
    EXTRA = 2

    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells    = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker  = np.full(num_cells, EXTRA, dtype = np.int32)

    # Get all facets
    inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside)
    cell_marker[inner_cells] = INTRA

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking boundaries of a unit square mesh with an interior square defined on [0.25, 0.75]^2
    
    The boundaries have the following tags:
        - tag value 3 : outer boundary (\partial\Omega) 
        - tag value 4 : interface gamma between inner and outer square
        - tag value 5 : interior facets

    """    
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.75), y_range)

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.25), y_range)

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.25), x_range)

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.75), x_range)

    # Tag values
    PARTIAL_OMEGA = 3
    GAMMA         = 4
    DEFAULT       = 5

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = GAMMA

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = GAMMA

    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = GAMMA

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = GAMMA

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def mark_boundaries_square_MMS(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit square mesh 
        with an interior square defined on [0.25, 0.75]^2.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.75), y_range)

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.25), y_range)

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.25), x_range)

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.75), x_range)

    # Tag values
    DEFAULT = 7
    PARTIAL_OMEGA   = 8
    LEFT    = 1
    RIGHT   = 2
    BOTTOM  = 3
    TOP     = 4

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = LEFT

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = RIGHT

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = TOP

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = BOTTOM

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def mark_subdomains_cube(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking subdomains of a unit cube mesh with an interior cube defined on [0.25, 0.75]^3.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y, z) = [0.25, 0.75]^3
        - tag value 2 : outer square, (x, y, z) = [0, 1]^3 \ [0.25, 0.75]^3
    
    """ 
    def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        """ Locator function for the inner square. """

        bool1 = np.logical_and(x[0] <= 0.75, x[0] >= 0.25) # True if inside inner box in x range
        bool2 = np.logical_and(x[1] <= 0.75, x[1] >= 0.25) # True if inside inner box in y range
        bool3 = np.logical_and(x[2] <= 0.75, x[2] >= 0.25) # True if inside inner box in z range
        
        inside_x_y =  np.logical_and(bool1, bool2)

        inside_cube = np.logical_and(bool3, inside_x_y)

        return inside_cube

    # Tag values
    INTRA = 1
    EXTRA = 2

    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells    = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker  = np.full(num_cells, EXTRA, dtype = np.int32)

    # Get all facets
    inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside)
    cell_marker[inner_cells] = INTRA

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_cube(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit cube mesh 
        with an interior cube defined on [0.25, 0.75]^3.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_right = np.isclose(x[0], 0.75)

        on_right = np.logical_and(x_right, np.logical_and(y_range, z_range))

        return on_right

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_left  = np.isclose(x[0], 0.25)

        on_left = np.logical_and(x_left, np.logical_and(y_range, z_range))

        return on_left

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_bottom = np.isclose(x[2], 0.25)
        
        on_bottom = np.logical_and(z_bottom, np.logical_and(x_range, y_range))

        return on_bottom

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_top    = np.isclose(x[2], 0.75)
        
        on_bottom = np.logical_and(z_top, np.logical_and(x_range, y_range))

        return on_bottom

    def front(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_front = np.isclose(x[1], 0.25)

        on_front = np.logical_and(y_front, np.logical_and(x_range, z_range))

        return on_front

    def back(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_back  = np.isclose(x[1], 0.75)

        on_back = np.logical_and(y_back, np.logical_and(x_range, z_range))

        return on_back

    # Tag values
    PARTIAL_OMEGA = 3
    GAMMA         = 4
    DEFAULT       = 5

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    # Get facets on the interface between the inner and outer cubes
    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = GAMMA

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = GAMMA

    front_facets = dfx.mesh.locate_entities(mesh, facet_dim, front)
    facet_marker[front_facets] = GAMMA

    back_facets = dfx.mesh.locate_entities(mesh, facet_dim, back)
    facet_marker[back_facets] = GAMMA

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = GAMMA

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = GAMMA

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def mark_boundaries_cube_MMS(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTags:
    """ Function for marking membrane and outer boundary of a unit cube mesh 
        with an interior cube defined on [0.25, 0.75]^3.

    """
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_right = np.isclose(x[0], 0.75)

        on_right = np.logical_and(x_right, np.logical_and(y_range, z_range))

        return on_right

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        x_left  = np.isclose(x[0], 0.25)

        on_left = np.logical_and(x_left, np.logical_and(y_range, z_range))

        return on_left

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_bottom = np.isclose(x[2], 0.25)
        
        on_bottom = np.logical_and(z_bottom, np.logical_and(x_range, y_range))

        return on_bottom

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range  = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        y_range  = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        z_top    = np.isclose(x[2], 0.75)
        
        on_bottom = np.logical_and(z_top, np.logical_and(x_range, y_range))

        return on_bottom

    def front(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_front = np.isclose(x[1], 0.25)

        on_front = np.logical_and(y_front, np.logical_and(x_range, z_range))

        return on_front

    def back(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        z_range = np.logical_and(x[2] >= 0.25, x[2] <= 0.75)
        y_back = np.isclose(x[1], 0.75)

        on_back = np.logical_and(y_back, np.logical_and(x_range, z_range))

        return on_back

    # Tag values
    DEFAULT = 7
    PARTIAL_OMEGA   = 8
    LEFT    = 1
    RIGHT   = 2
    FRONT   = 3
    BACK    = 4
    BOTTOM  = 5
    TOP     = 6

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    # Get facets on the interface between the inner and outer cubes
    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = LEFT

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = RIGHT

    front_facets = dfx.mesh.locate_entities(mesh, facet_dim, front)
    facet_marker[front_facets] = FRONT

    back_facets = dfx.mesh.locate_entities(mesh, facet_dim, back)
    facet_marker[back_facets] = BACK

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = TOP

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = BOTTOM

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

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