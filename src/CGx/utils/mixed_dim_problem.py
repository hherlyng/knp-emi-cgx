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
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley, IonicModel
from CGx.utils.misc  import flatten_list, mark_boundaries_cube_MMS, mark_boundaries_square_MMS, mark_subdomains_cube, mark_subdomains_square, range_constructor
from scipy.integrate import odeint, solve_ivp

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
        self.setup_boundary_conditions()
        if self.source_terms=="ion_injection": self.setup_source_terms()

        # Initialize empty ionic models list
        self.ionic_models = []
        
        print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds.\n")

    def read_config_file(self, config_file: yaml.__file__):
        
        yaml.add_constructor("!range", range_constructor) # Add reader for range of cell tags

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
            if 'T' in consts: self.T_value = consts['T']
            if 'R' in consts: self.R_value = consts['R']
            if 'F' in consts: self.F_value = consts['F']
            self.psi_value = self.R_value*self.T_value/self.F_value
        else:
            print("Setting all constants equal to 1.0.")
            self.T_value = self.R_value = self.F_value = self.psi_value = 1.0

        if 'C_M' in config:
            self.C_M_value = config['C_M']
        else:
            self.C_M_value = 1.0

        # Scaling mesh factor (default 1)
        if 'mesh_conversion_factor' in config: self.mesh_conversion_factor = float(config['mesh_conversion_factor'])

        # Finite element polynomial order (default 1)
        if 'fem_order' in config: self.fem_order = config['fem_order']

        # Boundary condition type (default pure Neumann BCs)
        if 'dirichlet_bcs' in config: self.dirichlet_bcs = config['dirichlet_bcs']

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
            self.ics_points = np.array(config['point_evaluation']['ics_points'])*self.mesh_conversion_factor
            self.ecs_points = np.array(config['point_evaluation']['ecs_points'])*self.mesh_conversion_factor
        else:
            self.point_evaluation = False

        if 'stimulus' in config:
            try:
                self.g_syn_bar_val = config['stimulus']['g_syn_bar']
                self.a_syn_val = config['stimulus']['a_syn']
                self.T_stim_val = config['stimulus']['T_stim']
            except:
                raise RuntimeError('For stimulus, provide g_syn_bar, a_syn and T in input file.')
        else:
            # Default stimulus of a single action potential with strength 40 S/m^2,
            self.g_syn_bar_val = 40.0
            self.a_syn_val = 1e-3
            self.T_stim_val = 1.0

        if 'stimulus_region' in config:
            self.stimulus_region = True
            self.stimulus_region_range = np.array(config['stimulus_region']['range'])*self.mesh_conversion_factor
            axes = {
                'x' : 0,
                'y' : 1,
                'z' : 2
            }
            self.stimulus_region_direction = axes[str(config['stimulus_region']['direction'])]
        else:
            self.stimulus_region = False

        if 'initial_conditions' in config:
            if 'filename' in config['initial_conditions']:
                # Initial conditions provided in a file
                self.initial_conditions = np.load(config['initial_conditions']['filename'])
            else:
                # Initial conditions provided as a dictionary in the config file
                self.initial_conditions = config['initial_conditions']
            self.find_initial_conditions = False # No need to find initial conditions
        else:
            self.find_initial_conditions = True # Need to find initial conditions
        
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
            self.boundary_tags = tags['boundary']
        else:
            print('Setting default: boundary tag = 1.')

        # Transform ints or lists to tuples
        if isinstance(self.intra_tags, int) or isinstance(self.intra_tags, list): self.intra_tags = tuple(self.intra_tags,)
        if isinstance(self.extra_tag, int) or isinstance(self.extra_tag, list): self.extra_tag = tuple(self.extra_tag,)
        if isinstance(self.boundary_tags, int) or isinstance(self.boundary_tags, list): self.boundary_tags = tuple(self.boundary_tags,)
        if isinstance(self.gamma_tags, int) or isinstance(self.gamma_tags, list): self.gamma_tags = tuple(self.gamma_tags,)
        if isinstance(self.glia_tags, int) or isinstance(self.glia_tags, list): self.glia_tags = tuple(self.glia_tags,)
        if isinstance(self.neuron_tags, int) or isinstance(self.neuron_tags, list): self.neuron_tags = tuple(self.neuron_tags,)

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
        
        ionic_tags = sorted(ionic_tags)
        gamma_tags = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags and not self.MMS_test:
            raise RuntimeError('Mismatch between membrane tags and ionic models tags.' \
                + f'\nIonic models tags: {ionic_tags}\nMembrane tags: {gamma_tags}')
        
        print('# Membrane tags = ', len(gamma_tags))
        print('# Ionic models  = ', len(self.ionic_models), '\n')

    def get_min_and_max_coordinates(self):
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
    
    def calculate_mesh_center(self):

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
            self.png_point = self.mesh.geometry.x[min_vertex]
            pprint("Phi m measurement point: ", self.png_point, flush=True)

            # Recast point array in a shape that enables point evaluation with scifem
            if self.mesh.geometry.dim==2:
                self.png_point = np.array([[self.png_point[0], self.png_point[1]]])
            else:
                self.png_point = np.array([self.png_point])
        
            # Broadcast to all processes
            self.comm.bcast(self.png_point, root=self.owner_rank_membrane_vertex)

    def setup_domain(self):

        print("Reading mesh from XDMF file...")

        # Get mesh and facet file names
        mesh_file = self.input_files['mesh_file']
        ft_file = self.input_files['facet_file']

        if not self.MMS_test:
            # Load mesh files with meshtags

            if mesh_file==ft_file:
                # Cell tags and facet tags in the same file
                with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                    # Read mesh and cell tags
                    self.mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode)
                    self.subdomains = xdmf.read_meshtags(self.mesh, name=self.ct_name)
                    self.subdomains.name = "ct"    

                    # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
                    self.mesh.topology.create_entities(self.mesh.topology.dim-1)
                    self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
                    self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

                    # Read facet tags
                    self.boundaries = xdmf.read_meshtags(self.mesh, name=self.ft_name)
                    self.boundaries.name = "ft" 
            
            else:
                # Cell tags and facet tags in separate files
                with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
                    # Read mesh and cell tags
                    self.mesh = xdmf.read_mesh(ghost_mode=self.ghost_mode)
                    self.subdomains = xdmf.read_meshtags(self.mesh, name=self.ct_name)
                    self.subdomains.name = "ct"

                # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
                self.mesh.topology.create_entities(self.mesh.topology.dim-1)
                self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
                self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

                with dfx.io.XDMFFile(MPI.COMM_WORLD, ft_file, 'r') as xdmf:
                    # Read facet tags
                    self.boundaries = xdmf.read_meshtags(self.mesh, name=self.ft_name)
                    self.boundaries.name = "ft"      
            
            # Scale mesh coordinates
            self.mesh.geometry.x[:] *= self.mesh_conversion_factor
        
        else:

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

            # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
            self.mesh.topology.create_entities(self.mesh.topology.dim-1)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

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

        if self.glia_tags is not None:
            # Store the neuron and glia computational cells
            self.neuron_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.neuron_tags]))
            self.glia_cells = np.concatenate(([self.subdomains.find(tag) for tag in self.glia_tags]))

        #-------------------------------------------------#        
        # Find vertices for evaluating the membrane potential
        if self.MMS_test:
            gamma_facets = np.concatenate(([self.boundaries.find(tag) for tag in self.gamma_tags]))
            self.find_membrane_point_closest_to_centroid(gamma_facets)
        else:
            gamma_facets = self.boundaries.find(self.stimulus_tags[0])
            if not self.stimulus_region:
                self.find_membrane_point_closest_to_centroid(gamma_facets)
                self.gamma_points = self.png_point
            else:
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

                gamma_points = [global_gamma_points[0]]

                # Find more points "downstream" of the stimulus
                coords = self.mesh.geometry.x[:, stim_dir]

                coord_min = self.comm.allreduce(coords.min(), op=MPI.MIN)
                coord_max = self.comm.allreduce(coords.max(), op=MPI.MAX)

                step = 5*(stim_range[1] - stim_range[0])
                i = 1
                lower_threshold = stim_range[0] + i*step
                upper_threshold = stim_range[1] + i*step
                while lower_threshold < coord_max:
                    lower_threshold = stim_range[0] + i*step
                    upper_threshold = stim_range[1] + i*step
                    mask = np.logical_and(
                                        gamma_coords[:, stim_dir] > lower_threshold,
                                        gamma_coords[:, stim_dir] < upper_threshold
                                    )
                    # Filter membrane coordinates with the mask
                    filtered_gamma_coords = gamma_coords[mask]
                    
                    # Find point local to each process, set None if process
                    # does not own any vertices 
                    local_gamma_point = None
                    if len(filtered_gamma_coords)>0:
                        local_gamma_point = filtered_gamma_coords[0]

                    # Gather all points and filter out Nones
                    global_gamma_points = self.comm.allgather(local_gamma_point)
                    global_gamma_points = [point for point in global_gamma_points if point is not None]
                    if len(global_gamma_points)>0:
                        gamma_points.append(global_gamma_points[0])
                    else:
                        break # No more points found
                    
                    # Increment index
                    i += 1
                
                # Convert to numpy array
                self.gamma_points = np.array(gamma_points)
                self.png_point = np.array([self.gamma_points[0]])
                    
        # Initialize injection site properties
        if self.source_terms=="ion_injection":
            x_min, x_max, _ = self.get_min_and_max_coordinates()
            domain_scale = x_max - x_min
            delta = domain_scale / 10
            self.initialize_injection_site(delta=delta)

    def find_steady_state_initial_conditions(self):

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
        timestep = 1e-7 # [s]
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

        # ATP pump
        I_hat = 0.449 # Maximum pump strength [A/m^2]
        m_K = 2.0 # ECS K+ pump threshold [mM]
        m_Na = 7.7 # ICS Na+ pump threshold [mM]

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

        # Gating variables
        phi_m_0_gating = (phi_m_0 - phi_rest)*1e3 # Convert to mV 
        n_0 = alpha_n(phi_m_0_gating) / (alpha_n(phi_m_0_gating) + beta_n(phi_m_0_gating))
        m_0 = alpha_m(phi_m_0_gating) / (alpha_m(phi_m_0_gating) + beta_m(phi_m_0_gating))
        h_0 = alpha_h(phi_m_0_gating) / (alpha_h(phi_m_0_gating) + beta_h(phi_m_0_gating))

        # Nernst potential
        E = lambda z_k, c_ki, c_ke: R*T/(z_k*F) * np.log(c_ke/c_ki)

        # ATP current
        par_1 = lambda K_e: 1 + m_K / K_e
        par_2 = lambda Na_i: 1 + m_Na / Na_i
        I_ATP = lambda Na_i, K_e: I_hat / (par_1(K_e)**2 * par_2(Na_i)**3)

        # Cotransporter currents
        I_KCC2 = lambda K_i, K_e, Cl_i, Cl_e: S_KCC2 * np.log((K_e * Cl_e)/(K_i*Cl_i))
        I_NKCC1_n = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: S_NKCC1 * 1 / (1 + np.exp(16 - K_e)) * np.log((Na_e * K_e * Cl_e**2)/(Na_i * K_i * Cl_i**2))


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
            
            if self.mesh.geometry.dim==2:
                area_g *= 1e-6 # Convert from m to m^2 assuming 1 um depth
            
            if self.comm.rank==0:

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
                    I_Na = (
                            (g_Na_leak + g_Na_bar * m**3 * h) * (phi_m_ - E_Na)
                            + 3*I_ATP(Na_i_, K_e_)
                            - I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_)
                        )
                    I_K  = (
                            (g_K_leak + g_K_bar * n**4)* (phi_m_ - E_K)
                            - 2*I_ATP(Na_i_, K_e_)
                            - I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_)
                            + I_KCC2(K_i_, K_e_, Cl_i_, Cl_e_)
                        )
                    I_Cl = (
                            -g_Cl_leak * (phi_m_ - E_Cl)
                            + 2*I_NKCC1_n(Na_i_, Na_e_, K_i_, K_e_, Cl_i_, Cl_e_)
                            - I_KCC2(K_i_, K_e_, Cl_i_, Cl_e_)
                        )
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

                    return [rhs_phi, rhs_Na_i, rhs_Na_e, rhs_K_i, rhs_K_e, rhs_Cl_i, rhs_Cl_e, rhs_n, rhs_m, rhs_h]

                init = [phi_m_0, Na_i_0, Na_e_0, K_i_0, K_e_0, Cl_i_0, Cl_e_0, n_0, m_0, h_0]
                sol_ = init

                for t, dt in zip(times, np.diff(times)):
                
                    if t > 0:
                        init = sol_
                        
                    # Integrate ODE system
                    sol = solve_ivp(
                            lambda t, x: two_compartment_rhs(x, t, args=init[:-3]),
                            [t, t+dt],
                            init,
                            method='BDF',
                            rtol=1e-6,
                            atol=1e-9
                        )
                    
                    sol_ = sol.y[:, -1] # Update previous solution
                    
                    if np.allclose(sol.y[:, 0], sol_, rtol=1e-6):
                        # Current solution equals previous solution
                        print("Steady state reached.")
                        break

                    # Checks
                    if np.isclose(t, max_time):
                        print("Max time reached without finding steady state. Exiting.")
                        break

                    if any(np.isnan(sol_)):
                        print("NaN values in solution. Exiting.")
                        break

                phi_m_init_val = sol_[0]
                Na_i_init_val = sol_[1]
                Na_e_init_val = sol_[2]
                K_i_init_val = sol_[3]
                K_e_init_val = sol_[4]
                Cl_i_init_val = sol_[5]
                Cl_e_init_val = sol_[6]
                n_init_val = sol_[7]
                m_init_val = sol_[8]
                h_init_val = sol_[9]
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
            phi_m_0_g = self.phi_m_g_init.value # Glial [V]

            # Glial mechanisms
            # Kir-Na and Na/K pump mechanisms
            E_K_0 = E(z_K, K_i_0, K_e_0)
            A = 1 + np.exp(0.433)
            B = 1 + np.exp(-(0.1186 + E_K_0) / 0.0441)
            C = lambda delta_phi_K: 1 + np.exp((delta_phi_K + 0.0185)/0.0425)
            D = lambda phi_m: 1 + np.exp(-(0.1186 + phi_m)/0.0441)

            rho_pump = 1.12e-6	 # Maximum pump rate (mol/m**2 s)
            P_Na_i = 10          # [Na+]i threshold for Na+/K+ pump (mol/m^3)
            P_K_e  = 1.5         # [K+]e  threshold for Na+/K+ pump (mol/m^3)

            # Pump expression
            I_glia_pump = lambda Na_i, K_e: rho_pump*F * (1 / (1 + (P_Na_i/Na_i)**(3/2))) * (1 / (1 + P_K_e/K_e))

            # Inward-rectifying K channel function
            f_Kir = lambda K_e, K_e_0, delta_phi_K, phi_m: A*B/(C(delta_phi_K)*D(phi_m))*np.sqrt(K_e/K_e_0)

            # Cotransporter strength and current
            g_KCC1 = 7e-1 # [S / m^2]
            S_KCC1 = g_KCC1 * R*T / F
            I_KCC1 = lambda K_i, K_e, Cl_i, Cl_e: S_KCC1 * np.log((K_e * Cl_e) / (K_i * Cl_i))

            g_NKCC1_g = 2e-2 # [S / m^2]
            S_NKCC1_g = g_NKCC1_g * R*T / F
            I_NKCC1_g = lambda Na_i, Na_e, K_i, K_e, Cl_i, Cl_e: S_NKCC1_g * np.log((Na_e * K_e * Cl_e**2)/(Na_i * K_i * Cl_i**2))

            if self.comm.rank==0:

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
                    I_Na_n = (
                            (g_Na_leak + g_Na_bar * m**3 * h) * (phi_m_n_ - E_Na_n)
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
                             -g_Cl_leak * (phi_m_n_ - E_Cl_n)
                            + 2*I_NKCC1_n(Na_i_n_, Na_e_, K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                            - I_KCC2(K_i_n_, K_e_, Cl_i_n_, Cl_e_)
                        )
                    I_ion_n = I_Na_n + I_K_n + I_Cl_n # Total neuronal ionic current

                    # Glial mechanisms
                    # Calculate Nernst potentials
                    E_Na_g = E(z_Na, Na_i_g_, Na_e_)
                    E_K_g  = E(z_K, K_i_g_, K_e_)
                    E_Cl_g = E(z_Cl, Cl_i_g_, Cl_e_)
                    
                    # Calculate glial ionic currents
                    delta_phi_K = phi_m_g_ - E_K_g
                    I_Na_g = (
                            g_Na_leak_g * (phi_m_g_ - E_Na_g)
                            + 3*I_glia_pump(Na_i_g_, K_e_)
                            - I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                        )
                    I_K_g = (
                            g_K_leak_g * f_Kir(x[4], K_e_, delta_phi_K, phi_m_g_) * (phi_m_g_ - E_K_g)
                            - 2*I_glia_pump(Na_i_g_, K_e_)
                            - I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                            + I_KCC1(K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                        )
                    I_Cl_g = (
                             -g_Cl_leak_g * (phi_m_g_ - E_Cl_g)
                            + 2*I_NKCC1_g(Na_i_g_, Na_e_, K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                            - I_KCC1(K_i_g_, K_e_, Cl_i_g_, Cl_e_)
                        )
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
                    phi_m_0_g, Na_i_0, K_i_0, Cl_i_0, n_0, m_0, h_0,
                        ]
                sol_ = init

                for t, dt in zip(times, np.diff(times)):
                
                    if t > 0:
                        init = sol_
                        
                    # Integrate ODE system
                    sol = solve_ivp(
                            lambda t, x: three_compartment_rhs(x, t, args=init[:-3]),
                            [t, t+dt],
                            init,
                            method='BDF',
                            rtol=1e-6,
                            atol=1e-9
                        )
                    
                    sol_ = sol.y[:, -1] # Update previous solution
                    
                    if np.allclose(sol.y[:, 0], sol_, rtol=1e-6):
                        # Current solution equals previous solution
                        print("Steady state reached.")
                        break

                    # Checks
                    if np.isclose(t, max_time):
                        print("Max time reached without finding steady state. Exiting.")
                        break

                    if any(np.isnan(sol_)):
                        print("NaN values in solution. Exiting.")
                        break
                        
                phi_m_n_init_val = sol_[0]
                Na_i_n_init_val = sol_[1]
                Na_e_init_val = sol_[2]
                K_i_n_init_val = sol_[3]
                K_e_init_val = sol_[4]
                Cl_i_n_init_val = sol_[5]
                Cl_e_init_val = sol_[6]
                phi_m_g_init_val = sol_[7]
                Na_i_g_init_val = sol_[8]
                K_i_g_init_val = sol_[9]
                Cl_i_g_init_val = sol_[10]
                n_init_val = sol_[11]
                m_init_val = sol_[12]
                h_init_val = sol_[13]

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

        print("Steady-state initial conditions determined by solving ODE system.")
    
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