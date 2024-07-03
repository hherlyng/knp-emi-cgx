import os
import ufl
import time
import yaml
import collections.abc

import dolfinx as dfx

from abc            import ABC, abstractmethod
from mpi4py         import MPI
from petsc4py       import PETSc
from CGx.utils.misc import flatten_list, mark_boundaries_cube_MMS, mark_boundaries_square_MMS, mark_subdomains_cube, mark_subdomains_square, check_if_file_exists

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

        # Read configuration file and initialize
        self.read_config_file(config_file=config_file)
        self.init()

        # Perform FEM setup
        self.setup_domain()
        self.setup_spaces()
        self.setup_boundary_conditions()

        # Initialize time
        self.t  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.dt = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt))

        # Initialize empty ionic models list
        self.ionic_models = []

        print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds.\n")

    def read_config_file(self, config_file: yaml.__file__):
        
        # Read input yaml file
        with open(config_file, 'r') as file:
            try:
                # load config dictionary from .yaml file
                config = yaml.safe_load(file)
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
                raise ValueError('Output directory ' + self.output_dir + ' does not exist.')
        else:
            # Set output directory to current directory
            self.output_dir = './'
        
        if 'cell_tag_file' in config and 'facet_tag_file' in config:

            mesh_file   = input_dir + config['cell_tag_file'] # cell tag file is also the mesh file
            facet_file = input_dir + config['facet_tag_file']

            # Check that the files exist
            [check_if_file_exists(file) for file in [mesh_file, facet_file]]

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
        if 'mesh_conversion_factor' in config: self.mesh_conversion_factor = config['mesh_conversion_factor']

        # Finite element polynomial order (default 1)
        if 'fem_order' in config: self.fem_order = config['fem_order']

        # Boundary condition type (default pure Neumann BCs)
        if 'dirichlet_bcs' in config: self.dirichlet_bcs = config['dirichlet_bcs']

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
            print('Using default ionic species: {Na, K, Cl}.')

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
            ionic_tags.append(model.tags)
        
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
            self.mesh.geometry.x[:] *= self.mesh_conversion_factor
        
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