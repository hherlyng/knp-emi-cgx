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
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley, IonicModel
from CGx.utils.misc  import flatten_list, mark_boundaries_cube_MMS, mark_boundaries_square_MMS, mark_subdomains_cube, mark_subdomains_square, range_constructor

pprint = print
print = PETSc.Sys.Print # Automatically flushes output to stream in parallel

class MixedDimensionalProblem(ABC):

    ghost_mode = dfx.mesh.GhostMode.shared_facet

    def __init__(self, config_file: yaml.__file__):
        
        tic = time.perf_counter()

        self.comm = MPI.COMM_WORLD
        print("Reading input data from " + config_file)

        # Options for the ffcx optimization
        cache_dir       = '.cache_' + str(config_file)
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
        self.setup_boundary_conditions()
        if self.source_terms=="ion_injection": self.setup_source_terms()

        # Initialize empty ionic models list
        self.ionic_models = []
        
        print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds.\n")
        
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

    def read_config_file(self, config_file: yaml.__file__):
        
        yaml.add_constructor("!range", range_constructor) # Add reader for range of cell tags

        # Read input yaml file
        with open(config_file, 'r') as file:
            try:
                # load config dictionary from .yaml file
                config = yaml.load(file, Loader=yaml.FullLoader)
            except yaml.YAMLError as e:
                print(e)

        if 'solver' in config:
            self.solver_config: dict = config['solver']
        else:
            raise RuntimeError('Provide solver configuration in input file.')

        if 'input_dir' in config:
            input_dir: str = config['input_dir']
        else:
            # Input directory is assumed to be here
            input_dir = './'
        
        if 'output_dir' in config:
            self.output_dir: str = config['output_dir']

            # Create output directory if it does not exist
            if not os.path.isdir(self.output_dir):
                print('Output directory ' + self.output_dir + ' does not exist. Creating the directory .')
                pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            # Set output directory to a new folder in current directory
            if not os.path.isdir('./output'): os.mkdir('./output')
            self.output_dir = './output/'
        
        if 'cell_tag_file' in config and 'facet_tag_file' in config:

            mesh_file: str  = input_dir + config['cell_tag_file'] # cell tag file is also the mesh file
            facet_file: str = input_dir + config['facet_tag_file']

            # Check that the files exist
            if not os.path.exists(mesh_file):
                print(f'The mesh and cell tag file {mesh_file} does not exist. Provide a valid mesh file.')
            if not os.path.exists(facet_file):
                print(f'The mesh and cell tag file {mesh_file} does not exist. Provide a valid facet tag file.')

            # Initialize input files dictionary and set mesh and facet files
            self.input_files: dict[str, str] = {}
            self.input_files['mesh_file']  = mesh_file
            self.input_files['facet_file'] = facet_file

            # Set tag names
            if "square" in mesh_file or mesh_file==facet_file:
                # Tags are in separate grids in the mesh files
                self.ct_name = "ct"
                self.ft_name = "ft"
            else:
                # Tags are under same hierarchy as mesh
                self.ct_name = "mesh"
                self.ft_name = "mesh"
        else:
            raise RuntimeError('Provide cell_tag_file and facet_tag_file fields in input file.')

        if 'dt' in config:
            self.dt = float(config['dt'])
        else:
            raise RuntimeError('Provide dt (timestep size) field in input file.')
        
        if 'time_steps' in config:
            self.time_steps = int(config['time_steps'])
        elif 'T' in config:
            self.time_steps = int(float(config['T']) / float(config['dt']))
        else:
            raise RuntimeError('Provide final time T or time_steps field in input file.')

        # Set mesh tags
        tags: dict[str, list[int] | int] = {}
        if 'ics_tags' in config:
            tags['intra'] = config['ics_tags'] 
        else:
            raise RuntimeError('Provide ics_tags (intracellular space tags) field in input file.')
        
        if 'ecs_tags'      in config: tags['extra']    = config['ecs_tags']
        if 'boundary_tags' in config: tags['boundary'] = config['boundary_tags']
        if 'membrane_tags' in config: tags['membrane'] = config['membrane_tags']
        if 'stimulus_tags' in config:
            self.stimulus_tags: list[int] = config['stimulus_tags']
        else:
            # All cells are stimulated
            self.stimulus_tags: list[int] = tags['membrane']
        if 'glia_tags' in config:
            tags['glia'] = config['glia_tags']
            tags['neuron'] = [tag for tag in tags['intra'] if tag not in tags['glia']]
        else:
            print('Setting default: all cell tags = neuron tags')
            tags['neuron'] = tags['intra']

        # Parse the tags
        self.parse_tags(tags=tags)

        # Set physical parameters
        if 'physical_constants' in config:
            consts = config['physical_constants']
            if 'T' in consts: self.T_value: float = consts['T']
            if 'R' in consts: self.R_value: float = consts['R']
            if 'F' in consts: self.F_value: float = consts['F']
            self.psi_value: float = self.R_value*self.T_value/self.F_value
        else:
            print("Setting all constants equal to 1.0.")
            self.T_value = self.R_value = self.F_value = self.psi_value = 1.0

        if 'C_M' in config:
            self.C_M_value: float = config['C_M']
        else:
            self.C_M_value = 1.0

        # Scaling mesh factor (default 1)
        if 'mesh_conversion_factor' in config:
            self.mesh_conversion_factor = float(config['mesh_conversion_factor'])

        # Finite element polynomial order (default 1)
        if 'fem_order' in config:
            self.fem_order: int = config['fem_order']

        # Boundary condition type (default pure Neumann BCs)
        if 'dirichlet_bcs' in config:
            self.dirichlet_bcs: bool = config['dirichlet_bcs']

        # Verification test flag
        if 'MMS_test' in config: 
            self.MMS_test = True
            self.dirichlet_bcs = True
            try:
                self.N_mesh = config['MMS_test']['N_mesh']
            except:
                raise RuntimeError('For MMS test, provide number of mesh cells "N_mesh" in input file.')
            
            try:
                self.dim = config['MMS_test']['dim']
            except:
                raise RuntimeError('For MMS test, provide dimension "dim" in input file.')

        # Set electrical conductivities (for EMI)
        if 'sigma_i' in config: self.sigma_i: float = config['sigma_i']
        if 'sigma_e' in config: self.sigma_e: float = config['sigma_e']

        # Set ion-specific parameters (for KNP-EMI)
        if 'ion_species' in config:

            self.ion_list = []

            for ion in config['ion_species']:

                ion_dict   = {'name' : ion}
                ion_params = config['ion_species'][ion]

                # Perform sanity checks
                if 'valence' not in ion_params:
                    raise RuntimeError('Valence of ion ' + ion + ' must be provided.')
                if 'diffusivity' not in ion_params:
                    raise RuntimeError('Diffusivity of ion ' + ion + ' must be provided.')
                if 'initial' not in ion_params:
                    raise RuntimeError('Initial condition of ion ' + ion + ' must be provided.')

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
                    ion_dict['f_i'] = 0.0
                    ion_dict['f_e'] = 0.0
                
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
            self.ics_points = np.array(config['point_evaluation']['ics_points'])*self.mesh_conversion_factor
            self.ecs_points = np.array(config['point_evaluation']['ecs_points'])*self.mesh_conversion_factor
        else:
            self.point_evaluation = False

        if 'stimulus' in config:
            try:
                assert 'conductance' in config['stimulus'], 'Provide conductance dictionary in stimulus configuration in input file.'
                g_dict: dict[str, float] = config['stimulus']['conductance']
                self.g_syn_bar_val: float = g_dict['g_syn_bar']
                self.a_syn_val: float = config['stimulus']['a_syn']
                self.T_stim_val: float = config['stimulus']['T_stim']
            except:
                raise RuntimeError('For stimulus, provide g_syn_bar, a_syn and T_stim in input file.')
            if 'tau_syn_rise' in config['stimulus'] or 'tau_syn_decay' in config['stimulus']:
                try:
                    self.tau_syn_rise: float = config['stimulus']['tau_syn_rise']
                    self.tau_syn_decay: float = config['stimulus']['tau_syn_decay']
                except:
                    raise RuntimeError('For rise and decay stimulus, provide tau_syn_rise and tau_syn_decay in input file.')
            if 'scale' in config['stimulus']:
                # Scale stimulus by the surface area of the stimulated membrane
                self.scale_stimulus = config['stimulus']['scale']
            else:
                raise RuntimeError('Provide whether to scale stimulus strength by surface area in stimulus configuration in input file.')

            self.g_Na_bar_val: float = g_dict.get('g_Na_bar', 1200.0) # Na max conductivity [S/m**2]
            self.g_K_bar_val: float  = g_dict.get('g_K_bar', 360.0) # K max conductivity [S/m**2]
            self.g_Na_leak_val: float   = g_dict.get('g_Na_leak', 0.3) # Neuronal Na leak conductivity [S/m**2]
            self.g_Na_leak_g_val: float = g_dict.get('g_Na_leak_g', 1.0) # Glial Na leak conductivity [S/m**2]
            self.g_K_leak_val: float    = g_dict.get('g_K_leak', 0.1) # Neuronal K leak conductivity [S/m**2]
            self.g_K_leak_g_val: float  = g_dict.get('g_K_leak_g', 16.96) # Glial K leak conductivity [S/m**2]
            self.g_Cl_leak_val: float   = g_dict.get('g_Cl_leak', 0.25) # Neuronal Cl leak conductivity [S/m**2]
            self.g_Cl_leak_g_val: float = g_dict.get('g_Cl_leak_g', 2.0) # Glial Cl leak conductivity [S/m**2]
        else:
            # Default stimulus of a single action potential with strength 40 S/m^2,
            self.g_syn_bar_val = 40.0
            self.a_syn_val     = 1e-3
            self.T_stim_val    = 1.0
            self.scale_stimulus = False
            self.g_Na_bar_val = 1200 # Na max conductivity [S/m**2]
            self.g_K_bar_val   = 360 # K max conductivity [S/m**2]
            self.g_Na_leak_val   = 1.0 # Neuronal Na leak conductivity [S/m**2]
            self.g_Na_leak_g_val = 1.0 # Glial Na leak conductivity [S/m**2]
            self.g_K_leak_val    = 4.0 # Neuronal K leak conductivity [S/m**2]
            self.g_K_leak_g_val  = 16.96 # Glial K leak conductivity [S/m**2]
            self.g_Cl_leak_val   = 0.25 # Neuronal Cl leak conductivity [S/m**2]
            self.g_Cl_leak_g_val = 0.50 # Glial Cl leak conductivity [S/m**2]

        if 'stimulus_region' in config:
            self.stimulus_region = True
            self.stimulus_region_range = np.array(config['stimulus_region']['range'])*self.mesh_conversion_factor
            axes = {
                'x' : 0,
                'y' : 1,
                'z' : 2
            }
            self.stimulus_region_direction: int = axes[str(config['stimulus_region']['direction'])]
        else:
            self.stimulus_region = False

        if 'initial_conditions' in config:
            # Initial conditions provided as a dictionary in the config file
            self.initial_conditions: dict[str, float] = config['initial_conditions']
            self.find_initial_conditions = False # No need to find steady-state initial conditions
        else:
            self.find_initial_conditions = True # Need to find steady-state initial conditions

        if 'membrane_data_tag' in config:
            # Tag for which membrane data is written to file
            self.membrane_data_tag: int = int(config['membrane_data_tag'])
        else:
            if len(self.stimulus_tags)>0:
                # If stimulus tags are present, record data in the first one
                self.membrane_data_tag = self.stimulus_tags[0]
            else:
                # Otherwise, record data in the first membrane tag
                self.membrane_data_tag = self.gamma_tags[0]
        
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
            self.intra_tags: list[int] | int = tags['intra']
        else:
            raise ValueError('Intra tag has to be provided.')
        
        if 'extra' in tags_set:
            self.extra_tag: int = tags['extra']
        else:
            print('Setting default: extra tag = 1.')
            self.extra_tag = 1
        
        if 'membrane' in tags_set:
            self.gamma_tags: list[int] | int = tags['membrane']
        else:
            print('Setting default: membrane tag = intra tag.')
            self.gamma_tags: list[int] | int = self.intra_tags

        if 'glia' in tags_set:
            self.glia_tags: list[int] | int = tags['glia']
            if len(self.glia_tags)==0:
                self.glia_flag = False
            else:
                self.glia_flag = True
        else:
            self.glia_tags = None
            self.glia_flag = False
        
        self.neuron_tags: list[int] | int = tags['neuron']
        
        if 'boundary' in tags_set:
            self.boundary_tags: list[int] | int = tags['boundary']
        else:
            print('Setting default: boundary tag = 1.')

        # Transform ints or lists to tuples
        self.intra_tags = tuple(self.intra_tags,)
        self.extra_tag = tuple(self.extra_tag,)
        self.boundary_tags = tuple(self.boundary_tags,)
        self.gamma_tags = tuple(self.gamma_tags,)
        self.neuron_tags = tuple(self.neuron_tags,)
        self.stimulus_tags = tuple(self.stimulus_tags,)
        if self.glia_flag: self.glia_tags = tuple(self.glia_tags,)

    def init_ionic_models(self, ionic_models: IonicModel | list[IonicModel]):

        self.ionic_models = ionic_models
        self.gating_variables = False # Initialize with no gating variables in the models

        # Initialize list
        ionic_tags = set()
    
        # Check that all intracellular space tags are present in some ionic model
        for model in self.ionic_models:

            model._init()

            for tag in model.tags:
                ionic_tags.add(tag)

            print("Added tags for ionic model: ", model.__str__())

            if isinstance(model, HodgkinHuxley):
                self.gating_variables = True
                print("Gating variables flag set to True.")
        
        ionic_tags: list[int] | int = sorted(ionic_tags)
        gamma_tags: list[int] | int = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags and not self.MMS_test and len(ionic_tags)!=0:
            raise RuntimeError('Mismatch between membrane tags and ionic models tags.' \
                + f'\nIonic models tags: {ionic_tags}\nMembrane tags: {gamma_tags}')
        
        print('# Membrane tags = ', len(gamma_tags))
        print('# Ionic models  = ', len(self.ionic_models), '\n')

    def get_min_and_max_coordinates(self) -> list[float]:
        if self.mesh.geometry.dim==3:
            xx, yy, zz = [self.mesh.geometry.x[:, i] for i in range(self.mesh.geometry.dim)]
            z_min = self.comm.allreduce(zz.min(), op=MPI.MIN)
            z_max = self.comm.allreduce(zz.max(), op=MPI.MAX)
        else:
            xx, yy = [self.mesh.geometry.x[:, i] for i in range(self.mesh.geometry.dim)]

        x_min = self.comm.allreduce(xx.min(), op=MPI.MIN)
        x_max = self.comm.allreduce(xx.max(), op=MPI.MAX)
        y_min = self.comm.allreduce(yy.min(), op=MPI.MIN)
        y_max = self.comm.allreduce(yy.max(), op=MPI.MAX)

        return [x_min, x_max, y_min, y_max, z_min, z_max] if self.mesh.geometry.dim==3 else [x_min, x_max, y_min, y_max]

    def calculate_mesh_center(self) -> np.ndarray:

        if self.mesh.geometry.dim==3:
            x_min, x_max, y_min, y_max, z_min, z_max = self.get_min_and_max_coordinates()
            z_c = (z_max + z_min) / 2
        else:
            x_min, x_max, y_min, y_max = self.get_min_and_max_coordinates()
            z_c = 0.0

        x_c = (x_max + x_min) / 2
        y_c = (y_max + y_min) / 2

        return np.array([x_c, y_c, z_c])

    def initialize_injection_site(self, delta: float):
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
        mesh_center = self.calculate_mesh_center()
        x_c, y_c, z_c = mesh_center
        
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

    def find_membrane_point_closest_to_centroid(self, gamma_facets: list | np.ndarray, within_stimulus_region: bool=False):
        """ Find the point on the largest cell's membrane
        that lies closest to the center point of the mesh.
        The membrane potential will be measured in this point.
        """
        # First, calculate the center point of the mesh
        mesh_center = self.calculate_mesh_center()

        # Get all membrane vertices of the given membrane facets
        gamma_vertices = dfx.mesh.compute_incident_entities(
                                                    self.mesh.topology,
                                                    gamma_facets,
                                                    self.mesh.topology.dim-1,
                                                    0
                                                )
        self.gamma_vertices = gamma_vertices
        num_local_gamma_vertices = self.mesh.topology.index_map(0).size_local
        gamma_vertices = np.unique(gamma_vertices)[gamma_vertices < num_local_gamma_vertices]
        gamma_coords = self.mesh.geometry.x[gamma_vertices]
        if within_stimulus_region:
            assert self.stimulus_region, print("No stimulus region defined.")
            # Pick one point within the stimulus region
            gamma_coords = gamma_coords[
                                np.logical_and(
                                    gamma_coords[:, self.stimulus_region_direction] > self.stimulus_region_range[0],
                                    gamma_coords[:, self.stimulus_region_direction] < self.stimulus_region_range[1]
                                )
                            ]
        
        # Find the vertex that lies closest to the cell's centroid
        distances = np.sum((gamma_coords - mesh_center)**2, axis=1)

        if self.comm.size==0:
            # Running in serial
            argmin_local = np.argmin(distances)
            min_vertex = gamma_vertices[argmin_local]
            png_point_ = self.mesh.geometry.x[min_vertex]
            pprint("Phi m measurement point: ", png_point_, flush=True)
            # Recast point array in a shape that enables point evaluation with scifem
            if self.mesh.geometry.dim==2:
                self.png_point = np.array([[png_point_[0], png_point_[1]]])
            else:
                self.png_point = np.array([png_point_])
        
        else:
            # Running in parallel: each MPI rank finds its local minimum
            # and the global minimum is found via MPI communication.
            # The global minimum point is broadcasted to all ranks.
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
                png_point_ = self.mesh.geometry.x[min_vertex]
                pprint("Phi m measurement point: ", png_point_, flush=True)
                
                # Recast point array in a shape that enables point evaluation with scifem
                if self.mesh.geometry.dim==2:
                    png_point = np.array([[png_point_[0], png_point_[1]]])
                else:
                    png_point = np.array([png_point_])
            else:
                png_point = None

            # Broadcast membrane point to all processes
            self.png_point = self.comm.bcast(png_point, root=self.owner_rank_membrane_vertex)
            self.png_dof   = self.comm.bcast(min_vertex, root=self.owner_rank_membrane_vertex)

    def setup_domain(self):

        print("Reading mesh from XDMF file...")

        # Get mesh and facet file names
        mesh_file: str = self.input_files['mesh_file']
        ft_file: str   = self.input_files['facet_file']

        if not self.MMS_test:
            # Load mesh files with meshtags

            if mesh_file==ft_file:
                # Cell tags and facet tags in the same file
                with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                    # Read mesh and cell tags
                    self.mesh: dfx.mesh.Mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode)
                    self.subdomains: dfx.mesh.MeshTags = xdmf.read_meshtags(self.mesh, name=self.ct_name)
                    self.subdomains.name = "ct"

                    # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
                    self.mesh.topology.create_entities(self.mesh.topology.dim-1)
                    self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
                    self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

                    # Read facet tags
                    self.boundaries: dfx.mesh.MeshTags = xdmf.read_meshtags(self.mesh, name=self.ft_name)
                    self.boundaries.name = "ft" 
            
            else:
                # Cell tags and facet tags in separate files
                with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                    # Read mesh and cell tags
                    self.mesh: dfx.mesh.Mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode)
                    self.subdomains: dfx.mesh.MeshTags = xdmf.read_meshtags(self.mesh, name=self.ct_name)
                    self.subdomains.name = "ct"

                # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
                self.mesh.topology.create_entities(self.mesh.topology.dim-1)
                self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
                self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

                with dfx.io.XDMFFile(MPI.COMM_WORLD, ft_file, 'r') as xdmf:
                    # Read facet tags
                    self.boundaries: dfx.mesh.MeshTags = xdmf.read_meshtags(self.mesh, name=self.ft_name)
                    self.boundaries.name = "ft"      
            
            # Scale mesh coordinates
            self.mesh.geometry.x[:] *= self.mesh_conversion_factor
        
        else:

            if self.dim==2:
                self.mesh: dfx.mesh.Mesh = dfx.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains: dfx.mesh.MeshTags = mark_subdomains_square(self.mesh)
                self.boundaries: dfx.mesh.MeshTags = mark_boundaries_square_MMS(self.mesh)
                self.gamma_tags = (1, 2, 3, 4)
            
            elif self.dim==3:
                self.mesh: dfx.mesh.Mesh = dfx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, nz=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains: dfx.mesh.MeshTags = mark_subdomains_cube(self.mesh)
                self.boundaries: dfx.mesh.MeshTags = mark_boundaries_cube_MMS(self.mesh)
                self.gamma_tags = (1, 2, 3, 4, 5, 6)

            # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
            self.mesh.topology.create_entities(self.mesh.topology.dim-1)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)
        
        # Create vertex to cell connectivity
        self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)

        # Generate integration entities for interior facet integrals
        # to ensure consistent direction of facet normal vector
        # across the cellular membranes
        subdomain_data_list = []
        for tag in self.gamma_tags:
            gamma_facets = self.boundaries.find(tag)
            gamma_integration_data = dfx.fem.compute_integration_domains(
                        dfx.fem.IntegralType.interior_facet,
                        self.mesh.topology,
                        gamma_facets,
                        self.boundaries.dim
                        )
            ordered_gamma_integration_data = gamma_integration_data.reshape(-1, 4).copy()
            if np.all(self.intra_tags < self.extra_tag):
                switch = self.subdomains.values[ordered_gamma_integration_data[:, 0]] \
                       > self.subdomains.values[ordered_gamma_integration_data[:, 2]]
            elif np.all(self.intra_tags > self.extra_tag):
                switch = self.subdomains.values[ordered_gamma_integration_data[:, 0]] \
                       < self.subdomains.values[ordered_gamma_integration_data[:, 2]]
            else:
                raise RuntimeError('Intracellular tags must be all smaller or all larger than extracellular tag.')
            
            if True in switch:
                ordered_gamma_integration_data[switch, :] = ordered_gamma_integration_data[switch][:, [2, 3, 0, 1]]
            subdomain_data_list.append((tag, ordered_gamma_integration_data.flatten()))
        
        # Integral measures for the domain
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains, metadata={"quadrature_degree":10}) # Volume integral measure
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=subdomain_data_list, metadata={"quadrature_degree":10}) # Facet integral measure

        if self.MMS_test:
            self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries) # Create boundary integral measure
            self.n_outer = ufl.FacetNormal(self.mesh) # Define outward normal on exterior boundary (\partial\Omega)

        # Store the neuron and glia computational cells
        self.neuron_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.neuron_tags]))
        if self.glia_flag:
            self.glia_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.glia_tags]))

        #-------------------------------------------------#        
        # Find vertices for evaluating the membrane potential
        
        if self.MMS_test:
            gamma_facets = np.concatenate(([self.boundaries.find(tag) for tag in self.gamma_tags]))
            self.find_membrane_point_closest_to_centroid(gamma_facets)
        else:
            gamma_facets = self.boundaries.find(self.membrane_data_tag)
            self.find_membrane_point_closest_to_centroid(gamma_facets)
            if self.stimulus_region:
                # Filter facets within stimulus region
                gamma_vertices = dfx.mesh.compute_incident_entities(
                                                                self.mesh.topology,
                                                                gamma_facets,
                                                                self.mesh.topology.dim-1,
                                                                0
                )
                num_local_gamma_vertices = self.mesh.topology.index_map(0).size_local
                gamma_vertices = np.unique(gamma_vertices)[gamma_vertices < num_local_gamma_vertices]
                gamma_coords = self.mesh.geometry.x[gamma_vertices]
                stim_dir = self.stimulus_region_direction
                stim_range = self.stimulus_region_range
                stimulus_region_mask = np.logical_and(
                                        gamma_coords[:, stim_dir] > stim_range[0],
                                        gamma_coords[:, stim_dir] < stim_range[1]
                                    )
                # Pick one point within the stimulus region
                filtered_gamma_coords = gamma_coords[stimulus_region_mask]
                # Find point local to each process, set None if process
                # does not own any vertices 
                local_gamma_point = None
                if len(filtered_gamma_coords)>0:
                    local_gamma_point = filtered_gamma_coords[0]

                # Gather all points and filter out Nones
                global_gamma_points = self.comm.allgather(local_gamma_point)
                global_gamma_points = [point for point in global_gamma_points if point is not None]

                self.gamma_points = np.array([global_gamma_points[0]])
            else:
                self.gamma_points = self.png_point
                    
        # Initialize injection site properties
        if self.source_terms=="ion_injection":
            x_min, x_max, _ = self.get_min_and_max_coordinates()
            domain_scale = x_max - x_min
            delta = domain_scale / 10
            self.initialize_injection_site(delta=delta)

    def calculate_compartment_volumes_and_surface_areas(self):
        """ Calculate the volumes [m^3] of the intra- and extracellular spaces
            and the surface areas [m^2] of the cellular membranes. 
        """
        
        self.vol_i_n = self.comm.allreduce(
                                dfx.fem.assemble_scalar(
                                    dfx.fem.form(1*self.dx(self.neuron_tags))
                                    ),
                                op=MPI.SUM
                            ) # [m^3]
        self.area_g_n = self.comm.allreduce(
                                dfx.fem.assemble_scalar(
                                    dfx.fem.form(1*self.dS(self.neuron_tags))
                                    ),
                                op=MPI.SUM
                            ) # [m^2]
        self.vol_e = self.comm.allreduce(
                                dfx.fem.assemble_scalar(
                                    dfx.fem.form(1*self.dx(self.extra_tag))
                                    ),
                                op=MPI.SUM
                            ) # [m^3]
        
        if self.glia_flag:
            self.vol_i_g = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dx(self.glia_tags))
                                        ),
                                    op=MPI.SUM
                                ) # [m^3]
            self.area_g_g = self.comm.allreduce(
                                    dfx.fem.assemble_scalar(
                                        dfx.fem.form(1*self.dS(self.glia_tags))
                                        ),
                                    op=MPI.SUM
                                ) # [m^2]