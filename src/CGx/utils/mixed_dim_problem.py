import os
import ufl
import time
import yaml
import pathlib
import collections.abc

import numpy   as np
import dolfinx as dfx

from abc            import ABC, abstractmethod
from mpi4py         import MPI
from petsc4py       import PETSc
from CGx.utils.misc import flatten_list, mark_boundaries_cube_MMS, mark_boundaries_square_MMS, mark_subdomains_cube, mark_subdomains_square, range_constructor

pprint = print
print = PETSc.Sys.Print

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

        # Perform FEM setup
        self.init()
        self.setup_spaces()
        self.setup_boundary_conditions()
        self.setup_source_terms()

        # Initialize time
        self.t  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.dt = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt))

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

        # Initial membrane potential (default -0.06774 Volts)
        if 'phi_M_init' in config: self.phi_M_init = config['phi_M_init']

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

    def parse_tags(self, tags: dict):

        allowed_tags = {'intra', 'extra', 'membrane', 'boundary'}

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
        
        if 'boundary' in tags_set:
            self.boundary_tag = tags['boundary']
        else:
            print('Setting default: boundary tag = 1.')

        # Transform ints or lists to tuples
        if isinstance(self.intra_tags, int) or isinstance(self.intra_tags, list): self.intra_tags = tuple(self.intra_tags,)
        if isinstance(self.extra_tag, int) or isinstance(self.extra_tag, list): self.extra_tag = tuple(self.extra_tag,)
        if isinstance(self.boundary_tag, int) or isinstance(self.boundary_tag, list): self.boundary_tag = tuple(self.boundary_tag,)
        if isinstance(self.gamma_tags, int) or isinstance(self.gamma_tags, list): self.gamma_tags = tuple(self.gamma_tags,)

    def init_ionic_model(self, ionic_models):

        self.ionic_models = ionic_models

        # Initialize list
        ionic_tags = []
    
        # Check that all intracellular space tags are present in some ionic model
        for model in self.ionic_models:
            for tag in model.tags:
                ionic_tags.append(tag)
        
        ionic_tags = sorted(flatten_list(ionic_tags))
        gamma_tags = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags:
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
                self.subdomains = xdmf.read_meshtags(self.mesh, name="mesh")
                self.subdomains.name = "ct"

            # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
            self.mesh.topology.create_entities(self.mesh.topology.dim-1)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)
            self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)

            with dfx.io.XDMFFile(MPI.COMM_WORLD, ft_file, 'r') as xdmf:
                # Read facet tags
                self.boundaries = xdmf.read_meshtags(self.mesh, name="mesh")
                self.boundaries.name = "ft"      
            
            # Scale mesh
            self.mesh.geometry.x[:] *= self.mesh_conversion_factor
        
        else:
            self.dim=2
            self.N_mesh = 32
            if self.dim==2:
                self.mesh = dfx.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_square(self.mesh)
                self.boundaries = mark_boundaries_square_MMS(self.mesh)
            
            elif self.dim==3:
                self.mesh = dfx.mesh.create_unit_cube(comm=MPI.COMM_WORLD, nx=self.N_mesh, ny=self.N_mesh, nz=self.N_mesh, ghost_mode=self.ghost_mode)
                self.subdomains = mark_subdomains_cube(self.mesh)
                self.boundaries = mark_boundaries_cube_MMS(self.mesh)

        # Integral measures for the domain
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains) # Volume integral measure
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.boundaries) # Facet integral measure

        # Find the point on the first cell's membrane
        # that lies closest to the center point of the mesh.
        # The membrane potential will be measured in this point.
        # First, calculate the center point of the mesh
        xx, yy, zz = [self.mesh.geometry.x[:, i] for i in range(self.mesh.geometry.dim)]
        x_min = self.comm.allreduce(xx.min(), op=MPI.MIN)
        x_max = self.comm.allreduce(xx.max(), op=MPI.MAX)
        y_min = self.comm.allreduce(yy.min(), op=MPI.MIN)
        y_max = self.comm.allreduce(yy.max(), op=MPI.MAX)
        z_min = self.comm.allreduce(zz.min(), op=MPI.MIN)
        z_max = self.comm.allreduce(zz.max(), op=MPI.MAX)
        x_c = (x_max + x_min) / 2
        y_c = (y_max + y_min) / 2
        z_c = (z_max + z_min) / 2
        mesh_center = np.array([x_c, y_c, z_c])

        # Find all membrane vertices of the cell
        gamma_facets = self.boundaries.find(self.gamma_tags[-1]) # Always take the tag of the largest cell
        gamma_vertices = dfx.mesh.compute_incident_entities(
                                                        self.mesh.topology,
                                                        gamma_facets,
                                                        self.mesh.topology.dim-1,
                                                        0
        )
        gamma_vertices = np.unique(gamma_vertices)
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
            delta = 1000*self.mesh_conversion_factor
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
