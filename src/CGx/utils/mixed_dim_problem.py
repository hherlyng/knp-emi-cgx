import os
import ufl
import time
import yaml
import pathlib
import collections.abc
import numpy   as np
import dolfinx as dfx

from abc             import ABC, abstractmethod
from mpi4py          import MPI
from petsc4py        import PETSc
from dolfinx.fem     import Constant
from CGx.utils.misc  import flatten_list, mark_boundaries_cube_MMS, mark_boundaries_square_MMS, mark_subdomains_cube, mark_subdomains_square, range_constructor
from scipy.integrate import odeint

pprint = print
print = PETSc.Sys.Print # Automatically flushes output to stream in parallel

class MixedDimensionalProblem(ABC):

    ghost_mode = dfx.mesh.GhostMode.shared_facet

    def __init__(self, config_file: yaml.__file__):
        
        tic = time.perf_counter()

        self.comm = MPI.COMM_WORLD
        print("Reading input data from " + config_file)

        # Options for the ffcx optimization
        cache_dir       = '.cache'
        compile_options = ["-Ofast", "-march=native"]
        self.jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                                "cache_dir"               : cache_dir,
                                "cffi_libraries"          : ["m"]}

        # Read configuration file and setup mesh
        self.read_config_file(config_file=config_file)
        self.setup_domain()

        # Initialize time
        self.t  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.dt = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt))

        # Perform FEM setup
        self.setup_constants()
        self.setup_spaces()
        self.init()
        self.find_steady_state_initial_conditions()
        self.setup_boundary_conditions()
        if self.source_terms=="ion_injection": self.setup_source_terms()

        # Initialize empty ionic models list
        self.ionic_models = []

        print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds.\n")

    def read_config_file(self, config_file: yaml.__file__):
        
        yaml.add_constructor("!range", range_constructor) # Add reading range of cell tags

        # Read input yaml file
        with open(config_file, 'r') as file:
            try:
                # load config dictionary from .yaml file
                config = yaml.load(file, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                print(e)

        if 'input_dir' in config:
            input_dir = config['input_dir']
        else:
            # input directory is here

            input_dir = './'
        
        if 'output_dir' in config:
            self.output_dir = config['output_dir']
            if not os.path.isdir(self.output_dir):
                print('Output directory ' + self.output_dir + ' does not exist. Creating the directory .')
                pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            # Set output directory to a new folder in current directory
            if not os.path.isdir('./output'): os.mkdir('./output')
            self.output_dir = './output/'
        
        if 'cell_tag_file' in config and 'facet_tag_file' in config:

            mesh_file   = input_dir + config['cell_tag_file'] # cell tag file is also the mesh file
            facet_file = input_dir + config['facet_tag_file']

            # Check that the files exist
            if not os.path.exists(mesh_file):
                print(f'The mesh and cell tag file {mesh_file} does not exist. Provide a valid mesh file.')
            if not os.path.exists(facet_file):
                print(f'The mesh and cell tag file {mesh_file} does not exist. Provide a valid facet tag file.')

            # Initialize input files dictionary and set mesh and facet files
            self.input_files = dict()
            self.input_files['mesh_file']  = mesh_file
            self.input_files['facet_file'] = facet_file
        else:
            raise RuntimeError('Provide cell_tag_file and facet_tag_file fields in input file.')

        if 'dt' in config:
            self.dt = config['dt']
        else:
            raise RuntimeError('Provide dt (timestep size) field in input file.')
        
        if 'time_steps' in config:
            self.time_steps = config['time_steps']
        elif 'T' in config:
            self.time_steps = int(config['T'] / config['dt'])
        else:
            raise RuntimeError('Provide final time T or time_steps field in input file.')

        # Set mesh tags
        tags = dict()
        if 'ics_tags' in config:
            tags['intra'] = config['ics_tags'] 
        else:
            raise RuntimeError('Provide ics_tags (intracellular space tags) field in input file.')
        
        if 'ecs_tags'      in config: tags['extra']    = config['ecs_tags']
        if 'boundary_tags' in config: tags['boundary'] = config['boundary_tags']
        if 'membrane_tags' in config: tags['membrane'] = config['membrane_tags']
        if 'stimulus_tags' in config:
            self.stimulus_tags = config['stimulus_tags']
        else:
            # All cells are stimulated
            self.stimulus_tags = tags['membrane']
        if 'glia_tags' in config:
            tags['glia'] = config['glia_tags']
            tags['neuron'] = [tag for tag in config['membrane_tags'] if tag not in config['glia_tags']]

        # Parse the tags
        self.parse_tags(tags=tags)

        # Set physical parameters
        if 'physical_constants' in config:
            consts = config['physical_constants']
            if 'T' in consts: self.T = consts['T']
            if 'R' in consts: self.R = consts['R']
            if 'F' in consts: self.F = consts['F']
            self.psi = self.R*self.T/self.F

        if 'C_M' in config: self.C_M = config['C_M']

        # Scaling mesh factor (default 1)
        if 'mesh_conversion_factor' in config: self.mesh_conversion_factor = float(config['mesh_conversion_factor'])

        # Finite element polynomial order (default 1)
        if 'fem_order' in config: self.fem_order = config['fem_order']

        # Boundary condition type (default pure Neumann BCs)
        if 'dirichlet_bcs' in config: self.dirichlet_bcs = config['dirichlet_bcs']

        # Verification test flag
        if 'MMS_test' in config: self.MMS_test = config['MMS_test']

        # Initial membrane potential
        if 'phi_m_init' in config: self.phi_m_init = config['phi_m_init']

        # Set electrical conductivities (for EMI)
        if 'sigma_i' in config: self.sigma_i = config['sigma_i']
        if 'sigma_e' in config: self.sigma_e = config['sigma_e']

        # Set ion-specific parameters (for KNP-EMI)
        if 'ion_species' in config:

            self.ion_list = []

            for ion in config['ion_species']:

                ion_dict   = {'name' : ion}
                ion_params = config['ion_species'][ion]

                # Perform sanity checks
                if 'valence' not in ion_params:
                    raise RuntimeError('Valence of ' + ion + ' must be provided.')
                if 'diffusivity' not in ion_params:
                    raise RuntimeError('Diffusivity of ' + ion + ' must be provided.')
                if 'initial' not in ion_params:
                    raise RuntimeError('Initial condition of ' + ion + ' must be provided.')

                # Fill in ion dictionary
                ion_dict['z']  = ion_params['valence']
                ion_dict['Di'] = ion_params['diffusivity']
                ion_dict['De'] = ion_params['diffusivity']
                ion_dict['ki_init'] = ion_params['initial']['ics']
                ion_dict['ke_init'] = ion_params['initial']['ecs']

                if 'source' in ion_params:
                    ion_dict['f_i'] = ion_params['source']['ics']
                    ion_dict['f_e'] = ion_params['source']['ecs']
                else:
                    print('Source terms for ' + ion + ' set to zero, as none were provided in the input file.')
                    ion_dict['f_i'] = 0
                    ion_dict['f_e'] = 0
                
                self.ion_list.append(ion_dict)
            
            self.N_ions = len(self.ion_list)
        
        else:
            if config['problem_type']=='KNP-EMI':
                print('Using default ionic species: {Na, K, Cl}.')
        
        if 'source_terms' in config:
            self.source_terms = config['source_terms']
        else:
            self.source_terms = None

        if 'point_evaluation' in config:
            self.point_evaluation = True
            self.ics_point = np.array(config['point_evaluation']['ics_point'])*self.mesh_conversion_factor
            self.ecs_point = np.array(config['point_evaluation']['ecs_point'])*self.mesh_conversion_factor
        else:
            self.point_evaluation = False

    def parse_tags(self, tags: dict):

        allowed_tags = {'intra', 'extra', 'membrane', 'boundary', 'glia', 'neuron'}

        tags_set = set(tags.keys())

        # Perform sanity check
        if not tags_set.issubset(allowed_tags):
            raise ValueError(f'Mismatch in tags.\nAllowed tags: {allowed_tags}\nInput tags: {tags_set}')

        # Print cell tag info
        if isinstance(tags['intra'], collections.abc.Sequence):
            print(f"# Cell tags = {len(tags['intra'])}.")           
        else:           
            print("Single cell tag.")    

        if 'intra' in tags_set:
            self.intra_tags = tags['intra']
        else:
            raise ValueError('Intra tag has to be provided.')
        
        if 'extra' in tags_set:
            self.extra_tag = tags['extra']
        else:
            print('Setting default: extra tag = 1.')
            self.extra_tag = 1
        
        if 'membrane' in tags_set:
            self.gamma_tags = tags['membrane']
        else:
            print('Setting default: membrane tag = intra tag.')
            self.gamma_tags = self.intra_tags

        if 'glia' in tags_set:
            self.glia_tags = tags['glia']
            self.neuron_tags = [tag for tag in tags['membrane'] if tag not in tags['glia']]
        else:
            print('Setting default: all membrane tags = neuron tags')
            self.glia_tags = None
            self.neuron_tags = self.gamma_tags
        
        if 'boundary' in tags_set:
            self.boundary_tag = tags['boundary']
        else:
            print('Setting default: boundary tag = 1.')

        # Transform ints or lists to tuples
        if isinstance(self.intra_tags, int) or isinstance(self.intra_tags, list): self.intra_tags = tuple(self.intra_tags,)
        if isinstance(self.extra_tag, int) or isinstance(self.extra_tag, list): self.extra_tag = tuple(self.extra_tag,)
        if isinstance(self.boundary_tag, int) or isinstance(self.boundary_tag, list): self.boundary_tag = tuple(self.boundary_tag,)
        if isinstance(self.gamma_tags, int) or isinstance(self.gamma_tags, list): self.gamma_tags = tuple(self.gamma_tags,)
        if isinstance(self.glia_tags, int) or isinstance(self.glia_tags, list): self.glia_tags = tuple(self.glia_tags,)
        if isinstance(self.neuron_tags, int) or isinstance(self.neuron_tags, list): self.neuron_tags = tuple(self.neuron_tags,)

    def init_ionic_model(self, ionic_models):

        self.ionic_models = ionic_models
        self.gating_variables = False # Initialize with no gating variables in the models

        # Initialize list
        ionic_tags = []
    
        # Check that all intracellular space tags are present in some ionic model
        for model in self.ionic_models:
            model._init()
            for tag in model.tags:
                if tag not in ionic_tags:
                    ionic_tags.append(tag)
            print("Added tags for ionic model: ", model.__str__())

            if model.__str__()=="Hodgkin-Huxley":
                self.gating_variables = True
                print("Gating variables flag set to True.")
        
        ionic_tags = sorted(flatten_list(ionic_tags))
        gamma_tags = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags and not self.MMS_test:
            raise RuntimeError('Mismatch between membrane tags and ionic models tags.' \
                + f'\nIonic models tags: {ionic_tags}\nMembrane tags: {gamma_tags}')
        
        
        print('# Membrane tags = ', len(gamma_tags))
        print('# Ionic models  = ', len(self.ionic_models), '\n')

    def setup_domain(self):

        print("Reading mesh from XDMF file...")

        # Rename file for readability
        mesh_file = self.input_files['mesh_file']
        ft_file = self.input_files['facet_file']

        if not self.MMS_test:
            # Load mesh files with meshtags
            
            with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                # Read mesh and cell tags
                self.mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode)
                self.subdomains = xdmf.read_meshtags(self.mesh, name="ct")
                self.subdomains.name = "ct"

            # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
            self.mesh.topology.create_entities(self.mesh.topology.dim-1)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

            with dfx.io.XDMFFile(MPI.COMM_WORLD, ft_file, 'r') as xdmf:
                # Read facet tags
                self.boundaries = xdmf.read_meshtags(self.mesh, name="ft")
                self.boundaries.name = "ft"      
            
            # Scale mesh
            self.mesh.geometry.x[:] *= self.mesh_conversion_factor
        
        else:
            self.dim=2
            self.N_mesh = 16
            if self.dim==2:
                self.mesh = dfx.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_square(self.mesh)
                self.boundaries = mark_boundaries_square_MMS(self.mesh)
                self.gamma_tags = (1, 2, 3, 4)
            
            elif self.dim==3:
                self.mesh = dfx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, nz=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_cube(self.mesh)
                self.boundaries = mark_boundaries_cube_MMS(self.mesh)
                self.gamma_tags = (1, 2, 3, 4, 5, 6)

            self.boundary_tag = 8

        # Integral measures for the domain
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains) # Volume integral measure
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.boundaries) # Facet integral measure

        if self.glia_tags is not None:
            # Store the neuron and glia computational cells
            self.neuron_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.neuron_tags]))
            self.glia_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.glia_tags]))

        # Find the point on the largest cell's membrane
        # that lies closest to the center point of the mesh.
        # The membrane potential will be measured in this point.
        # First, calculate the center point of the mesh
        if self.mesh.geometry.dim==3:
            xx, yy, zz = [self.mesh.geometry.x[:, i] for i in range(self.mesh.geometry.dim)]
            z_min = self.comm.allreduce(zz.min(), op=MPI.MIN)
            z_max = self.comm.allreduce(zz.max(), op=MPI.MAX)
            z_c = (z_max + z_min) / 2
        else:
            xx, yy = [self.mesh.geometry.x[:, i] for i in range(self.mesh.geometry.dim)]
            z_c = 0.0
        x_min = self.comm.allreduce(xx.min(), op=MPI.MIN)
        x_max = self.comm.allreduce(xx.max(), op=MPI.MAX)
        y_min = self.comm.allreduce(yy.min(), op=MPI.MIN)
        y_max = self.comm.allreduce(yy.max(), op=MPI.MAX)
        
        x_c = (x_max + x_min) / 2
        y_c = (y_max + y_min) / 2
        mesh_center = np.array([x_c, y_c, z_c])

        # Find all membrane vertices of the cell
        gamma_facets = self.boundaries.find(4)
        # gamma_facets = self.boundaries.find(68) # Always take the tag of the largest cell #10m, 100c stimulated in 66
        # gamma_facets = self.boundaries.find(89) # Always take the tag of the largest cell #20m, 100c stimulated in 88
        # gamma_facets = self.boundaries.find(self.gamma_tags[-1]) # Always take the tag of the largest cell
        gamma_vertices = dfx.mesh.compute_incident_entities(
                                                        self.mesh.topology,
                                                        gamma_facets,
                                                        self.mesh.topology.dim-1,
                                                        0
        )
        num_local_gamma_vertices = self.mesh.topology.index_map(0).size_local
        gamma_vertices = np.unique(gamma_vertices)[gamma_vertices < num_local_gamma_vertices]
        gamma_coords = self.mesh.geometry.x[gamma_vertices]

        # Find the vertex that lies closest to the cell's centroid
        distances = np.sum((gamma_coords - mesh_center)**2, axis=1)
        if len(distances)>0:
            # Rank has points to evaluate
            argmin_local = np.argmin(distances)
            min_dist_local = distances[argmin_local]
            min_vertex = gamma_vertices[argmin_local]
        else:
            # Set distance to infinity and placeholder vertex
            min_dist_local = np.inf
            min_vertex = -1
        # Communicate to find the vertex with minimal distance
        # from the center point
        min_eval = (min_dist_local, self.comm.rank)
        reduced = self.comm.allreduce(min_eval, op=MPI.MINLOC)
        self.owner_rank_membrane_vertex = reduced[1]

        if self.comm.rank==self.owner_rank_membrane_vertex:
            # Set the measurement point
            self.min_point = self.mesh.geometry.x[min_vertex]
            pprint("Phi m measurement point: ", self.min_point, flush=True)

            # Find the cell that contains the vertex and store
            # the owning process
            bb_tree = dfx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
            cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, self.min_point)
            colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, self.min_point)
            cc = colliding_cells.links(0)[0]
            self.membrane_cell = np.array(cc)
            
        if self.source_terms=="ion_injection":
            def injection_site_marker_function(x, tol=1e-14):
            
                lower_bound = lambda x, i, bound: x[i] >= bound - tol
                upper_bound = lambda x, i, bound: x[i] <= bound + tol

                return (
                    lower_bound(x, 0, self.x_L)
                    & lower_bound(x, 1, self.y_L)
                    & lower_bound(x, 2, self.z_L)
                    & upper_bound(x, 0, self.x_U)
                    & upper_bound(x, 1, self.y_U)
                    & upper_bound(x, 2, self.z_U)
                )

            # Initialize ion injection region 
            domain_scale = x_max - x_min
            delta = domain_scale / 10
            self.x_L = (x_c - delta) 
            self.y_L = (y_c - delta)
            self.z_L = (z_c - delta)
            self.x_U = (x_c + delta)
            self.y_U = (y_c + delta)
            self.z_U = (z_c + delta)

            # Find injection site cells and compute the
            # volume of that region
            self.injection_cells  = dfx.mesh.locate_entities(self.mesh,
                                                            self.mesh.topology.dim,
                                                            injection_site_marker_function)
            ct = dfx.mesh.meshtags(self.mesh,
                                self.mesh.topology.dim,
                                self.injection_cells,
                                np.full_like(self.injection_cells, 1, dtype=np.int32)
                                )
            self.dx_inj = ufl.Measure("dx", domain=self.mesh, subdomain_data=ct)
            self.injection_volume = self.comm.allreduce(
                                        dfx.fem.assemble_scalar(
                                            dfx.fem.form(
                                                1.0 * self.dx_inj(1)
                                            )
                                        ), 
                                        op=MPI.SUM
                                    )
        if self.point_evaluation:
            # Initialize cell placeholders
            self.ics_cell = np.array([], dtype=np.int32)
            self.ecs_cell = np.array([], dtype=np.int32)
            # Find cells for point evaluation of function
            bb_tree = dfx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
            cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, self.ics_point)
            colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, self.ics_point)
            if len(colliding_cells.links(0))>0:
                cc = colliding_cells.links(0)[0]
                self.ics_cell = np.array([cc], dtype=np.int32)
            cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, self.ecs_point)
            colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, self.ecs_point)
            if len(colliding_cells.links(0))>0:
                cc = colliding_cells.links(0)[0]
                self.ecs_cell = np.array([cc], dtype=np.int32)

    def find_steady_state_initial_conditions(self):
        
        print("Solving ODE system to find steady-state initial conditions ...")

        # Get constants
        R = self.R.value # Gas constant [J/(mol*K)]
        F = self.F.value # Faraday's constant [C/mol]
        T = self.T.value # Temperature [K]
        C_m = self.C_M.value # Membrane capacitance
        z_Na =  1 # Valence sodium
        z_K  =  1 # Valence potassium
        z_Cl = -1 # Valence chloride
        g_Na_bar  = self.g_Na_bar.value                 # Na max conductivity (S/m**2)
        g_K_bar   = self.g_K_bar.value                  # K max conductivity (S/m**2)    
        g_Na_leak = self.g_Na_leak.value              # Na leak conductivity (S/m**2) (Constant)
        g_Na_leak_g = self.g_Na_leak_g.value              # Na leak conductivity (S/m**2) (Constant)
        g_K_leak  = self.g_K_leak.value              # K leak conductivity (S/m**2)
        g_K_leak_g  = self.g_K_leak_g.value              # K leak conductivity (S/m**2)
        g_Cl_leak = self.g_Cl_leak.value                  # Cl leak conductivity (S/m**2) (Constant)
        g_Cl_leak_g = self.g_Cl_leak_g.value                  # Cl leak conductivity (S/m**2) (Constant)
        phi_rest = self.phi_rest.value  # Resting potential [V]

        # Define timespan for ODE solver
        timestep = 1e-6
        max_time = 1
        num_timesteps = int(max_time / timestep)
        times = np.linspace(0, max_time, num_timesteps+1)

        # Define initial condition guesses
        Na_i_0 = self.Na_i_init.value # [Mm]
        Na_e_0 = self.Na_e_init.value # [Mm]
        K_i_0 = self.K_i_init.value # [Mm]
        K_e_0 = self.K_e_init.value # [Mm]
        Cl_i_0 = self.Cl_i_init.value # [Mm]
        Cl_e_0 = self.Cl_e_init.value # [Mm]
        phi_m_0 = self.phi_m_init.value # [V]
        n_0 = 0.3 
        m_0 = 0.05
        h_0 = 0.65

        # ATP pump
        I_hat = 0.18 # Maximum pump strength [A/m^2]
        m_K = 3 # ECS K+ pump threshold [mM]
        m_Na = 12 # ICS Na+ pump threshold [mM]

        # Cotransporters
        S_KCC2 = 0.0034
        S_NKCC1 = 0.023

        # Hodgkin-Huxley parameters
        alpha_n = lambda V_m: 0.01e3 * (10.-V_m) / (np.exp((10. - V_m)/10.) - 1.)
        beta_n  = lambda V_m: 0.125e3 * np.exp(-V_m/80.)
        alpha_m = lambda V_m: 0.1e3 * (25. - V_m) / (np.exp((25. - V_m)/10.) - 1)
        beta_m  = lambda V_m: 4.e3 * np.exp(-V_m/18.)
        alpha_h = lambda V_m: 0.07e3 * np.exp(-V_m/20.)
        beta_h  = lambda V_m: 1.e3 / (np.exp((30. - V_m)/10.) + 1)

        # Nernst potential
        E = lambda z_k, c_ki, c_ke: R*T/(z_k*F) * np.log(c_ke/c_ki)

        # ATP current
        par_1 = lambda K_e: 1 + m_K/K_e
        par_2 = lambda Na_i: 1 + m_Na/Na_i
        I_ATP = lambda Na_i, K_e: I_hat / (par_1(K_e)**2 * par_2(Na_i)**3)

        # Cotransporter currents
        I_KCC2 = lambda K_i, K_e, Cl_i, Cl_e: S_KCC2 * np.log((K_i * Cl_i)/(K_e*Cl_e))
        I_NKCC1_n = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: S_NKCC1 * 1 / (1 + np.exp(16 - K_e)) * np.log((Na_i * K_i * Cl_i**2)/(Na_e * K_e * Cl_e**2))


        if self.glia_tags is None:
            # Only neuronal intracellular space
            vol_i = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.intra_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^3]
            vol_e = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.extra_tag))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^3]
            area_g = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dS(self.gamma_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^2]

            if self.comm.rank==0:

                print(f"{vol_i=}")
                print(f"{vol_e=}")
                print(f"{area_g=}")

                def two_compartment_rhs(x, t, args):
                    """ Right-hand side of ODE system for two-compartment system (neuron + ECS). """
                    # Extract gating variables at current timestep
                    n = x[7]; m = x[8]; h = x[9]

                    # Extract membrane potential and concentrations at previous timestep
                    phi_m_ = args[0]; Na_i_ = args[1]; Na_e_ = args[2]; K_i_ = args[3]; K_e_ = args[4]; Cl_i_ = args[5]; Cl_e_ = args[6]
                    
                    # Define potential used in gating variable expressions
                    phi_m_gating = (phi_m_ - phi_rest)*1e3 # Relative potential with unit correction

                    # Calculate Nernst potentials
                    E_Na = E(z_Na, Na_i_, Na_e_)
                    E_K  = E(z_K, K_i_, K_e_)
                    E_Cl = E(z_Cl, Cl_i_, Cl_e_)

                    # Calculate ionic currents
                    I_Na = (g_Na_leak + g_Na_bar * m**3 * h) * (phi_m_ - E_Na) + 3*I_ATP(Na_i_, K_e_) + I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_)
                    I_K = (g_K_leak + g_K_bar * n**4)* (phi_m_ - E_K) - 2*I_ATP(Na_i_, K_e_) + I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_) + I_KCC2(K_i_, K_e_, Cl_i_, Cl_e_)
                    I_Cl = g_Cl_leak * (phi_m_ - E_Cl) - 2*I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_) - I_KCC2(K_i_, K_e_, Cl_i_, Cl_e_)
                    I_ion = I_Na + I_K + I_Cl # Total current

                    # Define right-hand expressions
                    rhs_phi = -1/C_m * I_ion
                    rhs_Na_i = -I_Na * area_g / vol_i
                    rhs_Na_e =  I_Na * area_g / vol_e
                    rhs_K_i = -I_K * area_g / vol_i
                    rhs_K_e =  I_K * area_g / vol_e
                    rhs_Cl_i = -I_Cl * area_g / vol_i
                    rhs_Cl_e =  I_Cl * area_g / vol_e
                    rhs_n = alpha_n(phi_m_gating) * (1 - n) - beta_n(phi_m_gating) * n
                    rhs_m = alpha_m(phi_m_gating) * (1 - m) - beta_m(phi_m_gating) * m
                    rhs_h = alpha_h(phi_m_gating) * (1 - h) - beta_h(phi_m_gating) * h

                    return [rhs_phi, rhs_Na_i, rhs_Na_e, rhs_K_i, rhs_K_e, -rhs_Cl_i, -rhs_Cl_e, rhs_n, rhs_m, rhs_h]

                init = [phi_m_0, Na_i_0, Na_e_0, K_i_0, K_e_0, Cl_i_0, Cl_e_0, n_0, m_0, h_0]
                sol_ = init

                for t, dt in zip(times, np.diff(times)):
                
                    if t > 0:
                        init = sol_[-1]

                    sol = odeint(lambda x, t: two_compartment_rhs(x, t, args=init[:-3]), init, [t, t+dt])

                    if np.allclose(sol[0], sol[-1], rtol=1e-12):
                        # Current solution equals previous solution
                        print("Steady state reached.")
                        break

                    sol_ = sol # Update initial condition

                    # Checks
                    if np.isclose(t, max_time):
                        print("Max time reached without finding steady state. Exiting.")
                        break

                    if any(np.isnan(sol[-1])):
                        print("NaN values in solution. Exiting.")
                        break

                sol = sol[-1]
                for i in range(len(sol)):
                    print(f"{sol[i]:.15f}")

                phi_m_init_val = sol[0]
                Na_i_init_val = sol[1]
                Na_e_init_val = sol[2]
                K_i_init_val = sol[3]
                K_e_init_val = sol[4]
                Cl_i_init_val = sol[5]
                Cl_e_init_val = sol[6]
                n_init_val = sol[7]
                m_init_val = sol[8]
                h_init_val = sol[9]
            else:
                # Placeholders on non-root processes
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
            phi_m_init_val = self.comm.bcast(phi_m_init_val, root=0)
            Na_i_init_val = self.comm.bcast(Na_i_init_val, root=0)
            Na_e_init_val = self.comm.bcast(Na_e_init_val, root=0)
            K_i_init_val = self.comm.bcast(K_i_init_val, root=0)
            K_e_init_val = self.comm.bcast(K_e_init_val, root=0)
            Cl_i_init_val = self.comm.bcast(Cl_i_init_val, root=0)
            Cl_e_init_val = self.comm.bcast(Cl_e_init_val, root=0)
            n_init_val = self.comm.bcast(n_init_val, root=0)
            m_init_val = self.comm.bcast(m_init_val, root=0)
            h_init_val = self.comm.bcast(h_init_val, root=0)

            self.phi_m_init = Constant(self.mesh, phi_m_init_val)
            self.Na_i_init = Constant(self.mesh, Na_i_init_val)
            self.Na_e_init = Constant(self.mesh, Na_e_init_val)
            self.K_i_init = Constant(self.mesh, K_i_init_val)
            self.K_e_init = Constant(self.mesh, K_e_init_val)
            self.Cl_i_init = Constant(self.mesh, Cl_i_init_val)
            self.Cl_e_init = Constant(self.mesh, Cl_e_init_val)
            self.n_init = Constant(self.mesh, n_init_val)
            self.m_init = Constant(self.mesh, m_init_val)
            self.h_init = Constant(self.mesh, h_init_val)

            # Update ion dictionaries
            self.ion_list[0]['ki_init'] = self.Na_i_init
            self.ion_list[0]['ke_init'] = self.Na_e_init
            self.ion_list[1]['ki_init'] = self.K_i_init
            self.ion_list[1]['ke_init'] = self.K_e_init
            self.ion_list[2]['ki_init'] = self.Cl_i_init
            self.ion_list[2]['ke_init'] = self.Cl_e_init

        else:
            # Both neuronal and glial intracellular space
            vol_i_n = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.neuron_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^3]
            vol_i_g = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.glia_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^3]
            area_g_n = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dS(self.neuron_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^2]
            area_g_g = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dS(self.glia_tags))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^2]
            vol_e = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.extra_tag))
                                        ),
                                        op=MPI.SUM
                                        ) # [m^3]


            # Membrane potential initial conditions
            phi_m_0_n = phi_m_0 # Neuronal
            phi_m_0_g = self.phi_m_init_g.value # Glial [V]

            # Glial mechanisms
            # Kir-Na and Na/K pump mechanisms
            E_K_0 = E(z_K, K_i_0, K_e_0)
            A = 1 + np.exp(0.433)
            B = 1 + np.exp(-(0.1186 + E_K_0) / 0.0441)
            C = lambda delta_phi_K: 1 + np.exp((delta_phi_K + 0.0185)/0.0425)
            D = lambda phi_m: 1 + np.exp(-(0.1186 + phi_m)/0.0441)

            rho_pump = self.rho_pump.value	 # Maximum pump rate (mol/m**2 s)
            P_Na_i = self.P_Na_i.value          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
            P_K_e  = self.P_K_e.value         # [K+]e  threshold for Na+/K+ pump (mol/m^3)

            # Pump expression
            I_glia_pump = lambda Na_i, K_e: rho_pump*F * (1 / (1 + (P_Na_i/Na_i)**(3/2))) * (1 / (1 + P_K_e/K_e))

            # Inward-rectifying K channel function
            f_Kir = lambda K_e, K_e_0, delta_phi_K, phi_m: A*B/(C(delta_phi_K)*D(phi_m))*np.sqrt(K_e/K_e_0)

            # Cotransporter strength and current
            g_KCC1 = 7e-1 # [S / m^2]
            S_KCC1 = g_KCC1 * R*T / F
            I_KCC1 = lambda K_i, K_e, Cl_i, Cl_e: S_KCC1 * np.log((K_i * Cl_i) / (K_e * Cl_e))

            g_NKCC1_g = 2e-2 # [S / m^2]
            S_NKCC1_g = g_NKCC1_g * R*T / F
            I_NKCC1_g = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: S_NKCC1_g * np.log((Na_i * K_i * Cl_i**2)/(Na_e * K_e * Cl_e**2))

            if self.comm.rank==0:

                print(f"{vol_i_n=}")
                print(f"{vol_i_g=}")
                print(f"{vol_e=}")
                print(f"{area_g_n=}")
                print(f"{area_g_g=}")

                def three_compartment_rhs(x, t, args):
                    """ Right-hand side of ODE system for three-compartment system (neuron + glia + ECS). """
                    # Extract gating variables at current timestep
                    n = x[11]; m = x[12]; h = x[13]

                    # Extract membrane potential and concentrations at previous timestep
                    phi_m_n_ = args[0]; Na_i_n_ = args[1]; Na_e_ = args[2]; K_i_n_ = args[3]; K_e_ = args[4]; Cl_i_n_ = args[5]; Cl_e_ = args[6]
                    phi_m_g_ = args[7]; Na_i_g_ = args[8]; K_i_g_ = args[9]; Cl_i_g_ = args[10]
                    
                    # Neuronal mechanisms
                    # Define potential used in gating variable expressions
                    phi_m_gating = (phi_m_n_ - phi_rest)*1e3 # Relative potential with unit correction

                    # Calculate Nernst potentials
                    E_Na_n = E(z_Na, Na_i_n_, Na_e_)
                    E_K_n  = E(z_K, K_i_n_, K_e_)
                    E_Cl_n = E(z_Cl, Cl_i_n_, Cl_e_)

                    # Calculate neuronal ionic currents
                    I_Na_n = (g_Na_leak + g_Na_bar * m**3 * h) * (phi_m_n_ - E_Na_n) + 3*I_ATP(Na_i_n_, K_e_) + I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                    I_K_n = (g_K_leak + g_K_bar * n**4)* (phi_m_n_ - E_K_n) - 2*I_ATP(Na_i_n_, K_e_) + I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_) + I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                    I_Cl_n = g_Cl_leak * (phi_m_n_ - E_Cl_n) - 2*I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_) - I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                    I_ion_n = I_Na_n + I_K_n + I_Cl_n # Total neuronal ionic current

                    # Glial mechanisms
                    # Calculate Nernst potentials
                    E_Na_g = E(z_Na, Na_i_g_, Na_e_)
                    E_K_g  = E(z_K, K_i_g_, K_e_)
                    E_Cl_g = E(z_Cl, Cl_i_g_, Cl_e_)
                    
                    # Calculate glial ionic currents
                    delta_phi_K = phi_m_g_ - E_K_g
                    I_Na_g = g_Na_leak_g * (phi_m_g_ - E_Na_g) + 3*I_glia_pump(Na_i_g_, K_e_) + I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                    I_K_g = g_K_leak_g * f_Kir(x[4], K_e_, delta_phi_K, phi_m_g_) * (phi_m_g_ - E_K_g) - 2*I_glia_pump(Na_i_g_, K_e_) + I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_) + I_KCC1(K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                    I_Cl_g = g_Cl_leak_g * (phi_m_g_ - E_Cl_g) - 2*I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_) - I_KCC1(K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                    I_ion_g = I_Na_g + I_K_g + I_Cl_g

                    # Define right-hand expressions
                    rhs_phi_n = -1/C_m * I_ion_n
                    rhs_Na_i_n = -I_Na_n * area_g_n / vol_i_n
                    rhs_Na_e_n =  I_Na_n * area_g_n / vol_e
                    rhs_K_i_n = -I_K_n * area_g_n / vol_i_n
                    rhs_K_e_n =  I_K_n * area_g_n / vol_e
                    rhs_Cl_i_n = -I_Cl_n * area_g_n / vol_i_n
                    rhs_Cl_e_n =  I_Cl_n * area_g_n / vol_e
                    rhs_phi_g = -1/C_m * I_ion_g
                    rhs_Na_i_g = -I_Na_g * area_g_g / vol_i_g
                    rhs_Na_e_g =  I_Na_g * area_g_g / vol_e
                    rhs_K_i_g = -I_K_g * area_g_g / vol_i_g
                    rhs_K_e_g =  I_K_g * area_g_g / vol_e
                    rhs_Cl_i_g = -I_Cl_g * area_g_g / vol_i_g
                    rhs_Cl_e_g =  I_Cl_g * area_g_g / vol_e
                    rhs_Na_e = rhs_Na_e_n + rhs_Na_e_g
                    rhs_K_e = rhs_K_e_n + rhs_K_e_g
                    rhs_Cl_e = rhs_Cl_e_n + rhs_Cl_e_g
                    rhs_n = alpha_n(phi_m_gating) * (1 - n) - beta_n(phi_m_gating) * n
                    rhs_m = alpha_m(phi_m_gating) * (1 - m) - beta_m(phi_m_gating) * m
                    rhs_h = alpha_h(phi_m_gating) * (1 - h) - beta_h(phi_m_gating) * h

                    return [
                        rhs_phi_n, rhs_Na_i_n, rhs_Na_e, rhs_K_i_n, rhs_K_e, -rhs_Cl_i_n, -rhs_Cl_e, # Neuronal variables
                        rhs_phi_g, rhs_Na_i_g, rhs_K_i_g, -rhs_Cl_i_g, # Glial variables
                        rhs_n, rhs_m, rhs_h # Gating variables
                            ]

                init = [
                    phi_m_0_n, Na_i_0, Na_e_0, K_i_0, K_e_0, Cl_i_0, Cl_e_0,
                    phi_m_0_g, Na_i_0, K_i_0, Cl_i_0, n_0, m_0, h_0,
                        ]
                sol_ = init

                for t, dt in zip(times, np.diff(times)):
                
                    if t > 0:
                        init = sol_[-1]

                    sol = odeint(lambda x, t: three_compartment_rhs(x, t, args=init[:-3]), init, [t, t+dt])

                    if np.allclose(sol[0], sol[-1], rtol=1e-12):
                        # Current solution equals previous solution
                        print("Steady state reached.")
                        break

                    sol_ = sol # Update initial condition

                    # Checks
                    if np.isclose(t, max_time):
                        print("Max time reached without finding steady state. Exiting.")
                        break

                    if any(np.isnan(sol[-1])):
                        print("NaN values in solution. Exiting.")
                        break

                sol = sol[-1]
                for i in range(len(sol)):
                    print(f"{sol[i]:.15f}")

                phi_m_n_init_val = sol[0]
                Na_i_n_init_val = sol[1]
                Na_e_init_val = sol[2]
                K_i_n_init_val = sol[3]
                K_e_init_val = sol[4]
                Cl_i_n_init_val = sol[5]
                Cl_e_init_val = sol[6]
                phi_m_g_init_val = sol[7]
                Na_i_g_init_val = sol[8]
                K_i_g_init_val = sol[9]
                Cl_i_g_init_val = sol[10]
                n_init_val = sol[11]
                m_init_val = sol[12]
                h_init_val = sol[13]

            else:
                # Placeholders on non-root processes
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
            self.phi_m_n_init = self.comm.bcast(Constant(self.mesh, phi_m_n_init_val), root=0)
            self.Na_i_n_init = self.comm.bcast(Constant(self.mesh, Na_i_n_init_val), root=0)
            self.Na_e_init = self.comm.bcast(Constant(self.mesh, Na_e_init_val), root=0)
            self.K_i_n_init = self.comm.bcast(Constant(self.mesh, K_i_n_init_val), root=0)
            self.K_e_init = self.comm.bcast(Constant(self.mesh, K_e_init_val), root=0)
            self.Cl_i_n_init = self.comm.bcast(Constant(self.mesh, Cl_i_n_init_val), root=0)
            self.Cl_e_init = self.comm.bcast(Constant(self.mesh, Cl_e_init_val), root=0)
            self.phi_m_g_init = self.comm.bcast(Constant(self.mesh, phi_m_g_init_val), root=0)
            self.Na_i_g_init = self.comm.bcast(Constant(self.mesh, Na_i_g_init_val), root=0)
            self.K_i_g_init = self.comm.bcast(Constant(self.mesh, K_i_g_init_val), root=0)
            self.Cl_i_g_init = self.comm.bcast(Constant(self.mesh, Cl_i_g_init_val), root=0)
            self.n_init = self.comm.bcast(Constant(self.mesh, n_init_val), root=0)
            self.m_init = self.comm.bcast(Constant(self.mesh, m_init_val), root=0)
            self.h_init = self.comm.bcast(Constant(self.mesh, h_init_val), root=0) 

            # Communicate initial values from root process
            phi_m_n_init_val = self.comm.bcast(phi_m_n_init_val, root=0)
            Na_i_n_init_val = self.comm.bcast(Na_i_n_init_val, root=0)
            Na_e_init_val = self.comm.bcast(Na_e_init_val, root=0)
            K_i_n_init_val = self.comm.bcast(K_i_n_init_val, root=0)
            K_e_init_val = self.comm.bcast(K_e_init_val, root=0)
            Cl_i_n_init_val = self.comm.bcast(Cl_i_n_init_val, root=0)
            Cl_e_init_val = self.comm.bcast(Cl_e_init_val, root=0)
            phi_m_g_init_val = self.comm.bcast(phi_m_g_init_val, root=0)
            Na_i_g_init_val = self.comm.bcast(Na_i_g_init_val, root=0)
            K_i_g_init_val = self.comm.bcast(K_i_g_init_val, root=0)
            Cl_i_g_init_val = self.comm.bcast(Cl_i_g_init_val, root=0)
            n_init_val = self.comm.bcast(n_init_val, root=0)
            m_init_val = self.comm.bcast(m_init_val, root=0)
            h_init_val = self.comm.bcast(h_init_val, root=0)

            self.phi_m_n_init = Constant(self.mesh, phi_m_n_init_val)
            self.Na_i_n_init = Constant(self.mesh, Na_i_n_init_val)
            self.Na_e_init = Constant(self.mesh, Na_e_init_val)
            self.K_i_n_init = Constant(self.mesh, K_i_n_init_val)
            self.K_e_init = Constant(self.mesh, K_e_init_val)
            self.Cl_i_n_init = Constant(self.mesh, Cl_i_n_init_val)
            self.Cl_e_init = Constant(self.mesh, Cl_e_init_val)
            self.phi_m_g_init = Constant(self.mesh, phi_m_g_init_val)
            self.Na_i_g_init = Constant(self.mesh, Na_i_g_init_val)
            self.K_i_g_init = Constant(self.mesh, K_i_g_init_val)
            self.Cl_i_g_init = Constant(self.mesh, Cl_i_g_init_val)
            self.n_init = Constant(self.mesh, n_init_val)
            self.m_init = Constant(self.mesh, m_init_val)
            self.h_init = Constant(self.mesh, h_init_val)


            # Update ion dictionaries
            self.ion_list[0]['ki_init_n'] = self.Na_i_n_init
            self.ion_list[0]['ki_init_g'] = self.Na_i_g_init
            self.ion_list[0]['ke_init'] = self.Na_e_init
            self.ion_list[1]['ki_init_n'] = self.K_i_n_init
            self.ion_list[1]['ki_init_g'] = self.K_i_g_init
            self.ion_list[1]['ke_init'] = self.K_e_init
            self.ion_list[2]['ki_init_n'] = self.Cl_i_n_init
            self.ion_list[2]['ki_init_g'] = self.Cl_i_g_init
            self.ion_list[2]['ke_init'] = self.Cl_e_init

        print("Steady-state initial conditions found.")
    
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
    
    @abstractmethod
    def setup_source_terms(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass

    @abstractmethod
    def setup_constants(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass