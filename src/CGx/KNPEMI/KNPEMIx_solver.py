import time
import scifem
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy             as np
import dolfinx           as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py         import MPI
from petsc4py       import PETSc
from CGx.utils.misc import dump
from CGx.utils.calc_fluxes import create_flux_forms, compute_fluxes
from CGx.KNPEMI.KNPEMIx_problem     import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley

pprint = print # Allows flushing from arbitrary rank
print = PETSc.Sys.Print # Automatically flushes output to stream in parallel

class SolverKNPEMI:

    def __init__(self,
                problem: ProblemKNPEMI,
                solver_config: dict):
        """ Constructor for the KNP-EMI solver class. 
            Initializes solver parameters and output options.
        """

        # Problem instance, MPI communicator and time steps
        self.problem         = problem                 # The KNP-EMI problem
        self.comm: MPI.Comm  = problem.comm            # MPI communicator
        self.time_steps: int = problem.time_steps      # Number of timesteps

        # Output options
        self.save_xdmfs = solver_config['output']['save_xdmf'] if 'save_xdmf' in solver_config['output'] else False              # Option to save .xdmf output 
        self.save_pngs  = solver_config['output']['save_pngs'] if 'save_pngs' in solver_config['output'] else False              # Option to save .png  output
        self.save_cpoints = solver_config['output']['save_cpoints'] if 'save_cpoints' in solver_config['output'] else False  # Option to save checkpoint points
        self.save_dat = solver_config['output']['save_dat'] if 'save_dat' in solver_config['output'] else False                 # Option to save .npy output
        self.save_mat   = solver_config['output']['save_mat'] if 'save_mat' in solver_config['output'] else False  # Option to save the system matrix
        if 'save_interval' in solver_config['output']:
            self.save_interval = solver_config['output']['save_interval']  # Interval for saving output
        self.out_file_prefix = problem.output_dir # The output file directory
        
        # PETSc solver options
        self.direct_solver = solver_config['direct']    # Set direct solver/iterative solver option
        self.view_input = solver_config['view_ksp'] # Option to view KSP object
        if 'ksp_settings' in solver_config:
            ksp_settings = solver_config['ksp_settings']
            if 'ksp_type' in ksp_settings:
                self.ksp_type = ksp_settings['ksp_type'] # KSP type
            if 'pc_type' in ksp_settings:
                self.pc_type  = ksp_settings['pc_type']  # Preconditioner type
            if 'ksp_rtol' in ksp_settings:
                self.ksp_rtol = float(ksp_settings['ksp_rtol']) # KSP relative tolerance

        # Initialize output files
        if self.save_xdmfs : self.init_xdmf_savefile()
        if self.save_pngs  : self.init_png_savefile()
        if self.save_cpoints : self.init_checkpoint_file()
        if problem.point_evaluation : self.init_data()

        # Perform only a single timestep when saving system matrix
        if self.save_mat: self.time_steps = 1

    def assemble(self):
	
        print("Assembling linear system ...")
        p = self.problem # For ease of notation
        
        # Clear system matrix and RHS vector values to avoid accumulation
        self.A.zeroEntries()
        with self.b.localForm() as loc: loc.set(0.0)

        # Assemble system
        multiphenicsx.fem.petsc.assemble_matrix_block(self.A, p.a, bcs=p.bcs, restriction=(p.restriction, p.restriction)) # Assemble DOLFINx matrix
        multiphenicsx.fem.petsc.assemble_vector_block(self.b, p.L, p.a, bcs=p.bcs, restriction=p.restriction) # Assemble RHS vector
        self.A.assemble() # Assemble PETSc matrix

    def assemble_preconditioner(self):
        """ Assemble the preconditioner matrix. """
        
        p = self.problem # For ease of notation
        print("Assembling preconditioner ...")
        if not p.dirichlet_bcs:
            # P_assembled = multiphenicsx.fem.petsc.assemble_matrix_block(p.P, bcs=p.bcs, restriction=(p.restriction, p.restriction))
            P_assembled = multiphenicsx.fem.petsc.assemble_matrix_block(p.P, bcs=[], restriction=(p.restriction, p.restriction))
        else:
            P_assembled = multiphenicsx.fem.petsc.assemble_matrix_block(p.P, bcs=p.bcs, restriction=(p.restriction, p.restriction))
        P_assembled.assemble()
        self.P_ = P_assembled

        if self.save_mat:
            if p.MMS_test:
                print("Saving Pmat_MMS ...")
                dump(self.P_, 'output/Pmat_MMS')
            else:
                print("Saving Pmat")
                dump(self.P_, 'output/Pmat')

    def setup_solver(self):
        
        p = self.problem # For ease of notation
        
        # Create system matrix and right-hand side vector
        self.A: PETSc.Mat = multiphenicsx.fem.petsc.create_matrix_block(p.a, restriction=(p.restriction, p.restriction))
        self.b: PETSc.Vec = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)

        # Create solution vector
        self.x: PETSc.Vec = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)

        # Configure Krylov solver
        self.ksp = PETSc.KSP().create(self.comm)
        opts     = PETSc.Options()

        if self.direct_solver:
            print("Using direct solver ...")
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType('mumps')
            opts.setValue('pc_factor_zeropivot', 1e-22)

        else:
            print("Setting up iterative solver ...")

            # Set initial conditions as initial guess for the solver
            # Start by setting the initial conditions in the solution functions
            for idx, ion in enumerate(p.ion_list):
                if not p.glia_flag:
                    # Only neuronal cells
                    # Set the array values at the subspace dofs 
                    p.wh[0][idx].x.array[:] = ion['ki_init'].value
                    p.wh[1][idx].x.array[:] = ion['ke_init'].value
                else:
                    # Both glia and neurons present
                    # Set the array values at the subspace dofs 
                    p.wh[0][idx].x.array[p.neuron_dofs] = ion['ki_init_n'].value
                    p.wh[0][idx].x.array[p.glia_dofs]   = ion['ki_init_g'].value
                    p.wh[1][idx].x.array[:] = ion['ke_init'].value
            
            # Add potential initial conditions
            if not p.glia_flag:
                p.wh[0][p.N_ions].x.array[:] = p.phi_m_init.value
                p.wh[1][p.N_ions].x.array[:] = 0.0
            else:
                p.wh[0][p.N_ions].x.array[p.neuron_dofs] = p.phi_m_n_init.value
                p.wh[0][p.N_ions].x.array[p.glia_dofs]   = p.phi_m_g_init.value
                p.wh[1][p.N_ions].x.array[:] = 0.0

            # Insert the initial guess values into the solution vector
            # x used when solving the equation Ax = b 
            dofmaps: list[multiphenicsx.fem.petsc.DofMap] = [V.dofmap for V in p.V_list] # The function space dofmaps
            init_guess_vectors: list[PETSc.Vec] = [function.x.petsc_vec for function in [*p.wh[0], *p.wh[1]]] # Vectors of initial guesses
            with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                self.x, dofmaps, p.restriction) as C_dd_wrapper:
                for C_dd_component_local, data_vector in zip(C_dd_wrapper, tuple(init_guess_vectors)):
                    with data_vector.localForm() as data_vector_local:
                        C_dd_component_local[:] = data_vector_local

            # Set the KSP and PC types
            self.ksp.setType(self.ksp_type)
            pc = self.ksp.getPC()
            pc.setType(self.pc_type)

            if self.pc_type=="fieldsplit":
            
                # Create global offsets for each block in the linear system
                offsets = []
                global_start = 0
                for V in p.V_list:
                    offsets.append(global_start) # The starting global index of the current block
                    global_start += V.dofmap.index_map.size_global # The global size of the current block

                # Get the range of owned rows of A on the current rank
                lo, hi = self.A.getOwnershipRange()

                # Owned global indices per block = offsets[i] + [0..size_global-1]; filter to [lo, hi)
                block_dofs = []
                for i, V in enumerate(p.V_list):
                    global_size = V.dofmap.index_map.size_global # The global size of the dofmap of V
                    global_idx = offsets[i] + np.arange(global_size, dtype=np.int64) # Global indices of block i
                    local = global_idx[(global_idx >= lo) & (global_idx < hi)] # Filter to locally owned rows
                    block_dofs.append(local)

                # ICS concentration dofs: blocks 0..N_ions-1
                ics_all = np.concatenate(block_dofs[:p.N_ions])

                # ECS concentration dofs: blocks (N_ions+1)..(2*N_ions)
                ecs_all = np.concatenate(block_dofs[p.N_ions+1:2*p.N_ions+1])

                # Electric potential dofs: blocks N_ions (phi_i) and 2*N_ions+1 (phi_e)
                phi_all = np.concatenate([block_dofs[p.N_ions], block_dofs[2*p.N_ions+1]])

                # Filter the dofs to locally owned rows
                ics_local = ics_all[(ics_all >= lo) & (ics_all < hi)]
                ecs_local = ecs_all[(ecs_all >= lo) & (ecs_all < hi)]
                phi_local = phi_all[(phi_all >= lo) & (phi_all < hi)]

                # Create PETSc Index Set objects using the sorted local dofs 
                is_ics = PETSc.IS().createGeneral(np.sort(ics_local).astype(np.int32))
                is_ecs = PETSc.IS().createGeneral(np.sort(ecs_local).astype(np.int32))
                is_phi = PETSc.IS().createGeneral(np.sort(phi_local).astype(np.int32))

                # Set the field splits on the KSP preconditioner object
                pc.setFieldSplitIS(("ics", is_ics), ("ecs", is_ecs), ("phi", is_phi))

                # Set the iterative solver and preconditioner type for the field splits
                ksp_solver = 'preonly' 
                P_inv      = 'hypre'

                opts.setValue('pc_fieldsplit_type', 'multiplicative')

                opts.setValue('fieldsplit_ics_ksp_type', ksp_solver)
                opts.setValue('fieldsplit_ecs_ksp_type', ksp_solver)
                opts.setValue('fieldsplit_phi_ksp_type', ksp_solver)

                opts.setValue('fieldsplit_ics_pc_type',  P_inv)				
                opts.setValue('fieldsplit_ecs_pc_type',  P_inv)
                opts.setValue('fieldsplit_phi_pc_type',  "lu")

                # Set Hypre BoomerAMG options for the concentration splits
                opts.setValue('fieldsplit_ics_pc_hypre_boomeramg_max_iter', self.max_amg_iter)
                opts.setValue('fieldsplit_ics_pc_hypre_boomeramg_strong_threshold', 0.5)
                opts.setValue('fieldsplit_ics_pc_hypre_boomeramg_coarsen_type', "HMIS")
                opts.setValue('fieldsplit_ics_pc_hypre_boomeramg_interp_type', "ext+i")
                opts.setValue('fieldsplit_ecs_pc_hypre_boomeramg_max_iter', self.max_amg_iter)
                opts.setValue('fieldsplit_ecs_pc_hypre_boomeramg_strong_threshold', 0.5)
                opts.setValue('fieldsplit_ecs_pc_hypre_boomeramg_coarsen_type', "HMIS")
                opts.setValue('fieldsplit_ecs_pc_hypre_boomeramg_interp_type', "ext+i")
            
            else:
                # Apply preconditioner options for single-field preconditioner
                if self.ksp_type=='hypre':
                    opts.setValue('pc_hypre_boomeramg_max_iter', self.max_amg_iter)
                    if self.problem.mesh.geometry.dim==3:
                        opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.75)
            
            # Set miscellaneous KSP options
            opts.setValue('ksp_converged_reason', None)
            opts.setValue('ksp_rtol',      self.ksp_rtol)
            opts.setValue('ksp_max_it',    self.ksp_max_it)
            opts.setValue('ksp_norm_type', self.norm_type)
            opts.setValue('ksp_initial_guess_nonzero',   self.nonzero_init_guess)                

            # vector to collect number of iterations
            self.iterations = []

            if self.view_input:
                opts.setValue('ksp_view', None)
            if self.verbose:   
                opts.setValue('ksp_monitor_true_residual', None)

        # Set the configured options
        self.ksp.setFromOptions()      

        # vectors to collect runtimes
        self.solve_time    = []
        self.assembly_time = []

    def create_and_set_nullspace(self):
        """ Create null space for the electric potential in the case that the potentials are 
        only determined up to a constant for the pure Neumann boundary conditions case. """

        print("Creating and setting null space ...")

        p = self.problem # For ease of notation

        ns_vec = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)

        dofmaps: list[multiphenicsx.fem.petsc.DofMap] = [V.dofmap for V in p.V_list] # The function space dofmaps
        functions = [dfx.fem.Function(V) for V in p.V_list] # Finite element functions in each space
        functions[p.N_ions].x.array[:] = 1.0 # Intracellular potential
        functions[2*p.N_ions+1].x.array[:] = 1.0 # Extracellular potential
        Cs: list[PETSc.Vec] = [function.x.petsc_vec for function in functions] # Vector of constants

        idx = 0
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            ns_vec, dofmaps, p.restriction) as C_dd_wrapper:
            for C_dd_component_local, data_vector in zip(C_dd_wrapper, tuple(Cs)):
                if idx==p.N_ions or idx==(2*p.N_ions+1):  # Only potentials
                    with data_vector.localForm() as data_vector_local:
                        C_dd_component_local[:] = data_vector_local
                idx += 1
        ns_vec.normalize()
        
        # Create the PETSc null space vector and check that it is a valid nullspace of A
        nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=self.comm)

        # Set the nullspace
        assert nullspace.test(self.A) # Check that the nullspace is created correctly
        self.A.setNullSpace(nullspace)
        self.A.setNearNullSpace(nullspace)
        nullspace.remove(self.b)

        if not self.direct_solver and self.use_P_mat:
            self.P_.setNearNullSpace(nullspace)

    def solve(self):
        """ Solve the KNP-EMI problem. """
        
        # Aliases
        p      = self.problem
        t      = p.t
        dt     = p.dt
        wh     = p.wh

        dofmaps: list[multiphenicsx.fem.petsc.DofMap] = [V.dofmap for V in p.V_list] # The function space dofmaps
        functions: list[dfx.fem.Function] = [*wh[0], *wh[1]] # Finite element functions in each space

        setup_timer = 0.0

        tic = time.perf_counter()

        # Perform solver setup
        self.setup_solver()
        setup_timer += self.comm.allreduce(time.perf_counter() - tic, op=MPI.MAX)

        # Assemble preconditioner if enabled
        if not self.direct_solver and self.use_P_mat:
            tic = time.perf_counter()
            p.setup_preconditioner(self.use_block_Jacobi)
            setup_timer += self.comm.allreduce(time.perf_counter() - tic, op=MPI.MAX)
            self.assemble_preconditioner()

        # Time-stepping
        for i in range(1, self.time_steps+1):

            # Update current time
            p.t.value += float(dt.value)

            # Print timestep and time
            print('\nTime step ', i)
            print('t (ms) = ', 1000 * float(t.value))               
    
            # Update ODE-based ionic models
            if p.gating_variables:
                for model in p.ionic_models:
                    if isinstance(model, HodgkinHuxley):
                        model.update_t_mod()
                        model.update_gating_variables()
                    
            # Assemble system matrix and RHS vector
            tic = time.perf_counter()
            self.assemble()

            # Time the assembly
            assembly_time     = time.perf_counter() - tic
            max_assembly_time = self.comm.allreduce(assembly_time, op=MPI.MAX)
            self.tot_assembly_time += max_assembly_time
            self.assembly_time.append(max_assembly_time)
            print(f"Time dependent assembly in {max_assembly_time:0.4f} seconds")

            if self.save_mat:
                if self.problem.MMS_test:
                    print("Saving Amat_MMS ...")
                    dump(self.A, 'output/Amat_MMS')
                else:
                    print("Saving Amat ...")
                    dump(self.A, 'output/Amat')
                exit()

            # Perform initial timestep setup
            if i==1:
                tic = time.perf_counter()
                # Finalize configuration of PETSc structures
                if self.direct_solver: 
                    self.ksp.setOperators(self.A)
                    if not p.dirichlet_bcs:
                        self.ksp.getPC().setFactorSetUpSolverType()
                        self.ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # Option to support solving a singular matrix
                        self.ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # Option to support solving a singular matrix
                else: 
                    # Set operators of iterative solver
                    self.ksp.setOperators(self.A, self.P_) if self.use_P_mat else self.ksp.setOperators(self.A)

                if not p.dirichlet_bcs:
                    # Handle the nullspace of the electric potentials in the case of 
                    # pure Neumann boundary conditions
                    self.create_and_set_nullspace()

                # Add contribution to setup time
                setup_timer += self.comm.allreduce(time.perf_counter() - tic, op=MPI.MAX)

            # Solve
            tic = time.perf_counter()
            self.ksp.solve(self.b, self.x)
            self.tot_its += self.ksp.its

            # Update ghost values
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Time the linear solve
            solver_time     = time.perf_counter() - tic
            max_solver_time = self.comm.allreduce(solver_time, op=MPI.MAX)
            self.tot_solver_time += max_solver_time
            self.solve_time.append(max_solver_time)
            print(f"Solved in {max_solver_time:0.4f} seconds")

            # Store number of iterations in solve if using an iterative solver
            if not self.direct_solver: self.iterations.append(self.ksp.getIterationNumber())
            
            # Extract sub-components of solution and store them in the solution functions wh
            with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(self.x, dofmaps, p.restriction) as ui_ue_wrapper:
                for ui_ue_wrapper_local, component in zip(ui_ue_wrapper, (functions)):
                    with component.x.petsc_vec.localForm() as component_local:
                        component_local[:] = ui_ue_wrapper_local
            
            # Ghost update each component after writing locally owned entries
            for component in functions:
                component.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                                  mode=PETSc.ScatterMode.FORWARD)

            # Compute phi_m_prev = phi_i - phi_e
            with wh[0][p.N_ions].x.petsc_vec.localForm() as phi_i_loc, \
                wh[1][p.N_ions].x.petsc_vec.localForm() as phi_e_loc, \
                p.phi_m_prev.x.petsc_vec.localForm() as phi_m_loc:
                phi_m_loc[:] = phi_i_loc - phi_e_loc

            # Ghost update phi_m_prev so any neighbor reads are consistent
            p.phi_m_prev.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                                 mode=PETSc.ScatterMode.FORWARD)

            # Write output to file and save png
            if self.save_xdmfs and (i % self.save_interval == 0) : self.save_xdmf()
            if self.save_cpoints and (i % self.save_interval == 0) : self.save_checkpoint(i)
            if self.save_pngs: self.save_png()
            if p.point_evaluation: self.save_data(i)

            if i == self.time_steps:

                if self.save_pngs:
                    self.print_figures()
                    print("\nPNG output saved in ", self.out_file_prefix)

                if self.save_xdmfs:
                    self.close_xdmf()
                    print("\nXDMF output saved in ", self.out_file_prefix)

                if self.save_cpoints:
                    print("\nCheckpoints saved in ", self.out_file_prefix)
                
                if self.save_dat:
                    self.export_data()
                    print("\nPoint data saved in ", self.out_file_prefix)
                
                print("\nTotal setup time:", setup_timer)
                print("Total assembly time:", sum(self.assembly_time))
                print("Total solve time:",    sum(self.solve_time))

                # Print solver info and problem info
                self.print_info()

            if self.problem.MMS_test:
                self.problem.print_errors()


    def print_info(self):
        """ Print info about the problem and the solver. """

        p = self.problem # For ease of notation

        # Get number of dofs local to each processor and sum to rank 0
        num_dofs: int = p.interior.index_map.size_local*p.num_variables + p.exterior.index_map.size_local*p.num_variables
        num_dofs = self.comm.allreduce(num_dofs, op=MPI.SUM)

        # Get number of mesh cells local to each processor and sum to rank 0
        # Important to subtract shared dofs on the interfaces (cellular membrane, gamma)
        num_cells: int = p.mesh.geometry.dofmap.__len__() - p.mesh.topology.interprocess_facets().__len__()
        num_cells = self.comm.allreduce(num_cells, op=MPI.SUM)

        # Print problem and solver information
        print("\n#------------ PROBLEM -------------#\n")
        print("MPI Size = ", self.comm.size)
        print("Input mesh = ", p.input_files['mesh_file'])
        print("Global # mesh cells = ", num_cells)
        print("System size (global # dofs) = ", self.A.size[0])
        print("FEM order = ", p.fem_order)
        print("# Time steps = ", self.time_steps)
        print("dt = ", float(p.dt.value))

        if p.dirichlet_bcs:
            print("Using Dirichlet BCs.")
        else:
            print("Using Neumann BCs.")
        
        print("\n#------------ SOLVER -------------#\n")
        if self.direct_solver:
            print("Using direct solver mumps.")
        else:
            print("Solver type: [" + self.ksp_type + "+" + self.pc_type + "]")
            print(f"Tolerance: {self.ksp_rtol:.2e}")
            
            if self.use_P_mat: print("Preconditioner matrix P enabled.")
            
            print('Average iterations: ' + str(sum(self.iterations)/len(self.iterations)))


    def init_png_savefile(self):
        """ Initialize output .png file. Find a unique dof that lies on a cellular membrane (gamma) at the interface
        between an intracellular and extracellular space, and use this dof as a point for plotting the membrane potential.
        
        If gating variables are part of the problem, the gating variables are also plotted.
        """

        p = self.problem # For ease of notation
        owner = p.owner_rank_membrane_vertex # The rank owning the membrane vertex
        
        if self.comm.rank==owner:
            self.out_v_string = self.out_file_prefix + 'v.png'
            self.v_t = []
            self.v_t.append(1000 * p.phi_m_prev.x.array[p.png_dof]) # Converted to mV
            
            if hasattr(p, 'n'):
                # Gating variables are a part of the problem
                self.n_t = []
                self.m_t = []
                self.h_t = []
                self.out_gate_string = self.out_file_prefix + 'gating.png'
                self.n_t.append(p.n.x.array[p.png_dof])
                self.m_t.append(p.m.x.array[p.png_dof])
                self.h_t.append(p.h.x.array[p.png_dof])
        else:
            pass
            
        if hasattr(p, 'stim_ufl_expr'):
            self.stim_t = []
            self.stim_current_form = dfx.fem.form(p.stim_ufl_expr * p.dS(p.stimulus_tags))
            stim_current: float = self.comm.allreduce(
                                        dfx.fem.assemble_scalar(self.stim_current_form),
                                        op=MPI.SUM
                                    )
            self.stim_t.append(stim_current)
            
    def save_png(self):
        """ Save data for the .png output of the membrane electric potential, and the gating variables if 
        these are a part of the problem. """

        p = self.problem
        owner = p.owner_rank_membrane_vertex

        if self.comm.rank==owner:
            self.v_t.append(1000 * p.phi_m_prev.x.array[p.png_dof]) # Converted to mV
            
            if hasattr(p, 'n'):
                self.n_t.append(p.n.x.array[p.png_dof])
                self.m_t.append(p.m.x.array[p.png_dof])
                self.h_t.append(p.h.x.array[p.png_dof])
        else:
            pass

        if hasattr(p, 'stim_ufl_expr'):
            stim_current: float = self.comm.allreduce(
                                        dfx.fem.assemble_scalar(self.stim_current_form),
                                        op=MPI.SUM
                                    )
            print(f"Stimulus current at png point: {stim_current:.2e}")
            self.stim_t.append(stim_current)

    def init_data(self):
        
        p = self.problem
        
        self.ics_point_values = np.zeros((self.time_steps+1, p.num_variables, len(p.ics_points)))
        self.ecs_point_values = np.zeros((self.time_steps+1, p.num_variables, len(p.ecs_points)))
        self.gamma_point_values = np.zeros((self.time_steps+1, len(p.gamma_points)))

        for j in range(p.num_variables):
            self.ics_point_values[0, j] = scifem.evaluate_function(p.wh[0][j], p.ics_points).T
            self.ecs_point_values[0, j] = scifem.evaluate_function(p.wh[1][j], p.ecs_points).T

        self.gamma_point_values[0] = scifem.evaluate_function(p.phi_m_prev, p.gamma_points).T

        self.flux_forms  = create_flux_forms(p)
        self.flux_values = np.zeros((self.time_steps+1, 2*p.N_ions)) # Fluxes for each ion

    def save_data(self, i: int):
        """ Save function values evaluated in two points (one in ICS and one in the ECS)
            at time index i. """

        p = self.problem

        for j in range(p.num_variables):
            self.ics_point_values[i, j] = scifem.evaluate_function(p.wh[0][j], p.ics_points).T
            self.ecs_point_values[i, j] = scifem.evaluate_function(p.wh[1][j], p.ecs_points).T

        self.gamma_point_values[i] = scifem.evaluate_function(p.phi_m_prev, p.gamma_points).T

        fluxes: np.ndarray[float] = compute_fluxes(self.flux_forms, self.comm)
        for idx, val in enumerate(fluxes):
            self.flux_values[i, idx] = val

    def print_figures(self):
        """ Output .png plot of:
        - the membrane potential
        - the runtime of the solver

        Further problem-dependent output:
        - the gating variables (if present in the ionic models)
        - the number of solver iterations at each timestep (if using an iterative solver)

        """

        # Aliases
        dt = float(self.problem.dt.value)
        time_steps: int = self.time_steps
        times = np.linspace(0, 1000 * time_steps * dt, time_steps + 1)

        png_owner = self.problem.owner_rank_membrane_vertex
        if self.comm.rank==png_owner:
            # Save plot of membrane potential
            fig, ax = plt.subplots()    
            ax.plot(times, np.array(self.v_t).flatten())
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Membrane potential [mV]')
            fig.savefig(self.out_v_string)

            # save plot of gating variables
            if hasattr(self.problem, 'n'):
                fig, ax = plt.subplots()
                ax.plot(times, np.array(self.n_t).flatten(), label='n')
                ax.plot(times, np.array(self.m_t).flatten(), label='m')
                ax.plot(times, np.array(self.h_t).flatten(), label='h')
                ax.set_xlabel('Time [ms]')
                ax.legend()
                fig.savefig(self.out_gate_string)

        if hasattr(self.problem, 'stim_ufl_expr'):
            fig, ax = plt.subplots()
            ax.plot(times, np.array(self.stim_t).flatten(), label='Stimulus')
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Stimulus [A/m^2]')
            fig.savefig(self.out_file_prefix + 'stimulus.png')

        if hasattr(self.problem, 'gamma_points'):
            # Plot membrane potential in points on gamma
            fig, ax = plt.subplots()
            ax.plot(times, 1e3*self.gamma_point_values[:, :], color='green') # Converted to mV

            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Membrane potential [mV]', color='green')
            ax.legend()

            fig.savefig(self.out_file_prefix + 'point_evaluation_phi_m.png')
        
        if self.problem.point_evaluation:

            # Sodium (Na+)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(times, self.ics_point_values[:, 0, :], label='ICS Na+', color='blue')
            ax2.plot(times, self.ecs_point_values[:, 0, :], '--', label='ECS Na+', color='red')

            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel('ICS Na+ Concentration [mM’', color='blue')
            ax2.set_ylabel('ECS Na+ Concentration [mM]', color='red')

            # Combine legends from both axes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            fig.savefig(self.out_file_prefix + 'point_evaluation_Na.png')

            # Potassium (K+)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(times, self.ics_point_values[:, 1, :], label='ICS K+', color='green')
            ax2.plot(times, self.ecs_point_values[:, 1, :], '--', label='ECS K+', color='orange')

            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel('ICS K+ Concentration [mM]', color='green')
            ax2.set_ylabel('ECS K+ Concentration [mM]', color='orange')

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            fig.savefig(self.out_file_prefix + 'point_evaluation_K.png')

            # Chloride (Cl-)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(times, self.ics_point_values[:, 2, :], label='ICS Cl-', color='purple')
            ax2.plot(times, self.ecs_point_values[:, 2, :], '--', label='ECS Cl-', color='brown')

            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel('ICS Cl- Concentration [mM]', color='purple')
            ax2.set_ylabel('ECS Cl- Concentration [mM]', color='brown')

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            fig.savefig(self.out_file_prefix + 'point_evaluation_Cl.png')

        # Save iteration history
        if not self.direct_solver:
            fig, ax = plt.subplots()
            ax.plot(self.iterations)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Number of iterations')
            fig.savefig(self.out_file_prefix + 'iterations.png')

        # save runtime data
        fig, ax = plt.subplots()
        ax.plot(self.assembly_time, label='assembly')
        ax.plot(self.solve_time, label='solve')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Time [s]')
        ax.legend()
        fig.savefig(self.out_file_prefix + 'timings.png')

    def init_xdmf_savefile(self):
        """ Initialize .xdmf files for writing output. The mesh with meshtags is written to file,
        and an output file for the solution functions for ion concentrations and electric potentials
        is initialized. """

        p = self.problem # For ease of notation

        # Write tag data
        filename: str = self.out_file_prefix + 'subdomains.xdmf'
        xdmf_file = dfx.io.XDMFFile(self.comm, filename, "w")
        xdmf_file.write_mesh(p.mesh)
        xdmf_file.write_meshtags(p.subdomains, p.mesh.geometry)
        xdmf_file.close()

        # Create solution file and write mesh to file
        filename: str = self.out_file_prefix + 'solution.xdmf'
        self.xdmf_file = dfx.io.XDMFFile(self.comm, filename, "w")
        self.xdmf_file.write_mesh(p.mesh)
        self.xdmf_file.write_meshtags(p.subdomains, p.mesh.geometry)
        self.output_filename: str = filename # Store the output filename for post-processing

        # Write solution functions to file
        for idx in range(p.num_variables):
            self.xdmf_file.write_function(p.wh[0][idx], float(p.t.value))
            self.xdmf_file.write_function(p.wh[1][idx], float(p.t.value))

    def save_xdmf(self):
        """ Write solution functions for ion concentrations and electric potentials to file. """

        for idx in range(self.problem.num_variables):
            self.xdmf_file.write_function(self.problem.wh[0][idx], float(self.problem.t.value))
            self.xdmf_file.write_function(self.problem.wh[1][idx], float(self.problem.t.value))
    
    def init_checkpoint_file(self):
        """ Initialize checkpointing of solution functions. """
        p = self.problem
        self.cpoint_filename: str = self.out_file_prefix + "checkpoints"
        a4d.write_mesh(self.cpoint_filename, p.mesh)
        a4d.write_meshtags(self.cpoint_filename, p.mesh, meshtags=p.subdomains)
        a4d.write_meshtags(self.cpoint_filename, p.mesh, meshtags=p.boundaries)

        # Write concentrations to file
        for idx in range(p.N_ions):
            a4d.write_function(filename=self.cpoint_filename, u=p.u_out_i[idx], time=0)
            a4d.write_function(filename=self.cpoint_filename, u=p.u_out_e[idx], time=0)
        
        # Write membrane potential and gating variables to file
        a4d.write_function(filename=self.cpoint_filename, u=p.phi_m_prev, time=0)
        a4d.write_function(filename=self.cpoint_filename, u=p.n, time=0)
        a4d.write_function(filename=self.cpoint_filename, u=p.m, time=0)
        a4d.write_function(filename=self.cpoint_filename, u=p.h, time=0)

        return
    
    def save_checkpoint(self, i: int):
        """ Write solution to checkpoint file. """
        p = self.problem

        # Write concentrations to file
        for idx in range(p.N_ions):
            a4d.write_function(filename=self.cpoint_filename, u=p.u_out_i[idx], time=i)
            a4d.write_function(filename=self.cpoint_filename, u=p.u_out_e[idx], time=i)
        
        # Write membrane potential to file
        a4d.write_function(filename=self.cpoint_filename, u=p.phi_m_prev, time=i)
        a4d.write_function(filename=self.cpoint_filename, u=p.n, time=i)
        a4d.write_function(filename=self.cpoint_filename, u=p.m, time=i)
        a4d.write_function(filename=self.cpoint_filename, u=p.h, time=i)

        return

    def close_xdmf(self):
        """ Close .xdmf files. """

        # Close the XDMF file
        self.xdmf_file.close()
        
        # Run XDMF parser to restructure the data for better visualization
        # restructure_xdmf.run(self.output_filename)

    def export_data(self):
        """ Write numpy data to file. """

        # Save membrane potential values
        if self.comm.rank==self.problem.owner_rank_membrane_vertex:
            np.save(self.problem.output_dir+"phi_m.npy", np.array(self.v_t))
        
        # Save gamma point evaluation values
        if hasattr(self.problem, 'gamma_points'):
            np.save(self.problem.output_dir+"gamma_point_values.npy", self.gamma_point_values)

        # Save point evaluation values
        if self.problem.point_evaluation:
            np.save(self.problem.output_dir+"ics_point_values.npy", self.ics_point_values)
            np.save(self.problem.output_dir+"ecs_point_values.npy", self.ecs_point_values)

        # Save stimulus current values
        if hasattr(self.problem, 'stim_ufl_expr'):
            np.save(self.problem.output_dir+"stimulus.npy", np.array(self.stim_t))

        # Save flux values
        np.save(self.problem.output_dir+"flux_values.npy", self.flux_values)

        # Save timings
        np.save(self.problem.output_dir+"assembly_time.npy", np.array(self.assembly_time))
        np.save(self.problem.output_dir+"solve_time.npy", np.array(self.solve_time))
        if not self.direct_solver:
            # Save iterations
            np.save(self.problem.output_dir+"iterations.npy", np.array(self.iterations))

    # Default iterative solver parameters
    ksp_rtol           = 1e-7
    ksp_max_it         = 100000	
    ksp_type           = 'gmres' 
    pc_type            = 'hypre'
    norm_type          = 'preconditioned'
    max_amg_iter       = 1
    use_P_mat          = True # use P as preconditioner?
    verbose            = False
    use_block_Jacobi   = True
    nonzero_init_guess = True

    # Default output save interval
    save_interval     = 20 # save every nth timestep

    # Iteration counter and time variables
    tot_its           = 0
    tot_assembly_time = 0
    tot_solver_time   = 0