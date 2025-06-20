import time
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy             as np
import dolfinx           as dfx
import matplotlib.pyplot as plt

from mpi4py         import MPI
from petsc4py       import PETSc
from CGx.utils      import restructure_xdmf
from CGx.utils.misc import dump
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI

print = PETSc.Sys.Print # Enables printing only on rank 0 when running in parallel

class SolverKNPEMI(object):

    def __init__(self, problem: ProblemKNPEMI, view_input, use_direct_solver: bool=True,
                 save_xdmfs: bool=False, save_pngs: bool=False, save_mat: bool=False):
        """ Constructor. """

        self.problem    = problem                 # The KNP-EMI problem
        self.comm       = problem.comm            # MPI communicator
        self.time_steps = problem.time_steps      # Number of timesteps
        self.direct_solver = use_direct_solver    # Set direct solver/iterative solver option
        self.save_xdmfs = save_xdmfs              # Option to save .xdmf output 
        self.save_pngs  = save_pngs               # Option to save .png  output
        self.save_mat   = save_mat                # Option to save the system matrix
        self.out_file_prefix = problem.output_dir # The output file directory
        self.view_input = view_input

        # Initialize varational form
        self.problem.setup_variational_form()

        # Initialize output files
        if save_xdmfs : self.init_xdmf_savefile()
        if save_pngs  : self.init_png_savefile()

        # Perform only a single timestep when saving system matrix
        if self.save_mat: self.time_steps = 1

    def assemble(self):
	
        print("Assembling linear system ...")
        p = self.problem # For ease of notation
        
        # Clear system matrix and RHS vector values to avoid accumulation
        self.A.zeroEntries()
        self.b.array[:] = 0

        # Assemble system
        multiphenicsx.fem.petsc.assemble_matrix_block(self.A, p.a, bcs=p.bcs, restriction=(p.restriction, p.restriction)) # Assemble DOLFINx matrix
        multiphenicsx.fem.petsc.assemble_vector_block(self.b, p.L, p.a, bcs=p.bcs, restriction=p.restriction) # Assemble RHS vector
        self.A.assemble() # Assemble PETSc matrix


    def assemble_preconditioner(self):
        """ Assemble the preconditioner matrix. """
        
        p = self.problem # For ease of notation

        if not p.dirichlet_bcs:
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
        self.A  = multiphenicsx.fem.petsc.create_matrix_block(p.a, restriction=(p.restriction, p.restriction))
        self.b  = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)
        
        # Create solution vector
        self.x = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)

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

            # set initial guess
            for idx, ion in enumerate(p.ion_list):
                # Get dof mapping between subspace and parent space
                _, sub_to_parent_i = p.wh[0].sub(idx).function_space.collapse()
                _, sub_to_parent_e = p.wh[1].sub(idx).function_space.collapse()

                # Set the array values at the subspace dofs 
                p.wh[0].sub(idx).x.array[sub_to_parent_i] = ion['ki_init']
                p.wh[1].sub(idx).x.array[sub_to_parent_e] = ion['ke_init']
            
            p.wh[0].sub(p.N_ions).x.array[:] = p.phi_i_init
            p.wh[1].sub(p.N_ions).x.array[:] = p.phi_e_init

            self.ksp.setType(self.ksp_type)
            pc = self.ksp.getPC()
            pc.setType(self.pc_type)

            if self.pc_type=="fieldsplit":

                # aliases
                Wi = p.W[0]
                We = p.W[1]

                is0 = PETSc.IS().createGeneral(Wi.sub(0).dofmap.list)
                is1 = PETSc.IS().createGeneral(Wi.sub(1).dofmap.list)
                is2 = PETSc.IS().createGeneral(Wi.sub(2).dofmap.list)
                is3 = PETSc.IS().createGeneral(Wi.sub(3).dofmap.list)
                is4 = PETSc.IS().createGeneral(We.sub(0).dofmap.list)
                is5 = PETSc.IS().createGeneral(We.sub(1).dofmap.list)
                is6 = PETSc.IS().createGeneral(We.sub(2).dofmap.list)
                is7 = PETSc.IS().createGeneral(We.sub(3).dofmap.list)

                fields = [('0', is0), ('1', is1), ('2', is2), ('3', is3),('4', is4), ('5', is5),('6', is6), ('7', is7)]
                pc.setFieldSplitIS(*fields)

                ksp_solver = 'preonly'
                P_inv      = 'hypre'

                opts.set('pc_fieldsplit_type', 'additive')

                opts.set('fieldsplit_0_ksp_type', ksp_solver)
                opts.set('fieldsplit_1_ksp_type', ksp_solver)
                opts.set('fieldsplit_2_ksp_type', ksp_solver)
                opts.set('fieldsplit_3_ksp_type', ksp_solver)
                opts.set('fieldsplit_4_ksp_type', ksp_solver)
                opts.set('fieldsplit_5_ksp_type', ksp_solver)
                opts.set('fieldsplit_6_ksp_type', ksp_solver)
                opts.set('fieldsplit_7_ksp_type', ksp_solver)

                opts.set('fieldsplit_0_pc_type',  P_inv)				
                opts.set('fieldsplit_1_pc_type',  P_inv)
                opts.set('fieldsplit_2_pc_type',  P_inv)
                opts.set('fieldsplit_3_pc_type',  P_inv)
                opts.set('fieldsplit_4_pc_type',  P_inv)
                opts.set('fieldsplit_5_pc_type',  P_inv)
                opts.set('fieldsplit_6_pc_type',  P_inv)
                opts.set('fieldsplit_7_pc_type',  P_inv)
            
            opts.setValue('ksp_converged_reason', None)
            opts.setValue('ksp_rtol',      self.ksp_rtol)
            opts.setValue('ksp_max_it',    self.ksp_max_it)
            opts.setValue('ksp_norm_type', self.norm_type)
            opts.setValue('ksp_initial_guess_nonzero',   self.nonzero_init_guess)
            if self.ksp_type=='hypre': opts.setValue('pc_hypre_boomeramg_max_iter', self.max_amg_iter)
            if self.problem.mesh.geometry.dim == 3: opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.5)

            # vector to collect number of iterations
            self.iterations    = []

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
        p = self.problem # For ease of notation

        ns_vec = multiphenicsx.fem.petsc.create_vector_block(p.L, restriction=p.restriction)

        # Get potential subspaces
        _, Vi_dofs = p.W[0].sub(p.N_ions).collapse()
        _, Ve_dofs = p.W[1].sub(p.N_ions).collapse()
        ci = dfx.fem.Function(p.V)
        ce = dfx.fem.Function(p.V)
        ci.x.array[Vi_dofs] = 1.0
        ce.x.array[Ve_dofs] = 1.0

        Ci = ci.x.petsc_vec
        Ce = ce.x.petsc_vec 

        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
            ns_vec, [p.W[0].dofmap, p.W[1].dofmap], p.restriction) as C_dd_wrapper:
            for C_dd_component_local, data_vector in zip(C_dd_wrapper, (Ci, Ce)):
                if data_vector is not None:  # skip third block
                    with data_vector.localForm() as data_vector_local:
                        C_dd_component_local[:] = data_vector_local
        ns_vec.normalize()
        
        # Create the PETSc nullspace vector and check that it is a valid nullspace of A
        nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=self.comm)
        assert nullspace.test(self.A) # Check that the nullspace is created correctly
        
        # Set the nullspace
        if self.direct_solver:
            self.A.setNullSpace(nullspace)
            nullspace.remove(self.b)
        else:
            self.A.setNullSpace(nullspace)
            self.A.setNearNullSpace(nullspace)
            nullspace.remove(self.b)

    def solve(self):

        # perform setup
        self.setup_solver()
        
        # Aliases
        p      = self.problem
        x      = self.x
        t      = p.t
        dt     = p.dt
        wh     = p.wh
        V, _   = p.V.sub(p.N_ions).collapse()

        # Previous membrane potentials
        phi_i_p = dfx.fem.Function(V)
        phi_e_p = dfx.fem.Function(V)

        # Assemble preconditioner if enabled
        if not self.direct_solver and self.use_P_mat:
            p.setup_preconditioner(self.use_block_Jacobi)
            self.assemble_preconditioner()
        
        setup_timer = 0

        # Time-stepping
        for i in range(self.time_steps):

            # Update current time
            p.t.value += float(dt.value)

            # Print timestep and time
            print('\nTime step ', i + 1)
            print('t (ms) = ', 1000 * float(t.value))               

            # Set up the variational form
            tic = time.perf_counter()
            p.setup_variational_form()
            setup_timer += self.comm.allreduce(time.perf_counter() - tic, op=MPI.MAX)

            # Assemble system matrix and RHS vector
            tic = time.perf_counter()
            self.assemble()

            # Time the assembly
            assembly_time     = time.perf_counter() - tic
            max_assembly_time = self.comm.allreduce(assembly_time, op=MPI.MAX)
            self.tot_assembly_time += max_assembly_time
            self.assembly_time.append(max_assembly_time)
            print(f"Time dependent assembly in {max_assembly_time:0.4f} seconds")

            # Perform initial timestep setup
            if i==0:
                tic = time.perf_counter()

                if not p.dirichlet_bcs:
                    # Handle the nullspace of the electric potentials in the case of 
                    # pure Neumann boundary conditions
                    self.create_and_set_nullspace()

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

                # Add contribution to setup time
                setup_timer += self.comm.allreduce(time.perf_counter() - tic, op=MPI.MAX)

            # Solve
            tic = time.perf_counter()
            self.ksp.solve(self.b, x)
            self.tot_its += self.ksp.its

            # Update ghost values
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Time the linear solve
            solver_time     = time.perf_counter() - tic
            max_solver_time = self.comm.allreduce(solver_time, op=MPI.MAX)
            self.tot_solver_time += max_solver_time
            self.solve_time.append(max_solver_time)
            print(f"Solved in {max_solver_time:0.4f} seconds")

            # Store number of iterations in solve if using an iterative solver
            if not self.direct_solver: self.iterations.append(self.ksp.getIterationNumber())
            
            # Extract sub-components of solution and store them in the solution functions wh
            with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [p.W[0].dofmap, p.W[1].dofmap], p.restriction) as ui_ue_wrapper:
                for ui_ue_wrapper_local, component in zip(ui_ue_wrapper, (wh[0], wh[1])):
                    with component.x.petsc_vec.localForm() as component_local:
                        component_local[:] = ui_ue_wrapper_local

            # Update previous timestep values
            p.u_p[0].x.array[:] = wh[0].x.array.copy() # Intracellular ions and potential
            p.u_p[1].x.array[:] = wh[1].x.array.copy() # Extracellular ions and potential
            phi_i_p.x.array[:]  = wh[0].sub(p.N_ions).collapse().x.array.copy() # Intracellular potential
            phi_e_p.x.array[:]  = wh[1].sub(p.N_ions).collapse().x.array.copy() # Extracellular potential
            p.phi_M_prev.x.array[:] = phi_i_p.x.array.copy() - phi_e_p.x.array.copy() # Membrane potential      

            # Write output to file and save png
            if self.save_xdmfs and (i % self.save_interval == 0) : self.save_xdmf()
            if self.save_pngs: self.save_png()

            if i == self.time_steps - 1:
                # Last timestep, consolidate output files
                if self.save_pngs:
                    self.print_figures()
                    print("\nPNG output saved in ", self.out_file_prefix)

                if self.save_xdmfs:
                    self.close_xdmf()
                    print("\nXDMF output saved in ", self.out_file_prefix)

                print("\nTotal setup time:", setup_timer)
                print("Total assembly time:", sum(self.assembly_time))
                print("Total solve time:",    sum(self.solve_time))

                # Print solver info and problem info
                self.print_info()
                      
            if self.save_mat:
                if self.problem.MMS_test:
                    print("Saving Amat_MMS ...")
                    dump(self.A, 'output/Amat_MMS')
                else:
                    print("Saving Amat ...")
                    dump(self.A, 'output/Amat')

    def print_info(self):
        """ Print info about the problem and the solver. """

        p = self.problem # For ease of notation

        # Get number of dofs local to each processor and sum to rank 0
        num_dofs = p.interior.index_map.size_local + p.exterior.index_map.size_local
        num_dofs = self.comm.allreduce(num_dofs, op=MPI.SUM)

        # Get number of mesh cells local to each processor and sum to rank 0
        # Important to subtract shared dofs on the interfaces (cellular membrane, gamma)
        num_cells = p.mesh.geometry.dofmap.__len__() - p.mesh.topology.interprocess_facets().__len__()
        num_cells = self.comm.allreduce(num_cells, op=MPI.SUM)

        # Print problem and solver information
        print("\n#------------ PROBLEM -------------#\n")
        print("MPI Size = ", self.comm.size)
        print("Input mesh = ", p.input_files['mesh_file'])
        print("Global # mesh cells = ", num_cells)
        print("Global # dofs = ", num_dofs)
        print("FEM order = ", p.fem_order)

        if not self.direct_solver: print("System size = ", self.A.size[0])
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
            
            if self.use_P_mat: print("Preconditioning enabled.")
            
            print('Average iterations: ' + str(sum(self.iterations)/len(self.iterations)))


    def init_png_savefile(self):
        """ Initialize output .png file. Find a unique dof that lies on a cellular membrane (gamma) at the interface
        between an intracellular and extracellular space, and use this dof as a point for plotting the membrane potential.
        
        If gating variables are a part of the problem, the gating variables are also plotted.
        """

        p = self.problem # For ease of notation

        phi_M_space = p.phi_M_prev.function_space # Membrane electric potential
        
        # Get indices of the membrane (gamma) facets that are being stimulated
        if len(p.stimulus_tags) > 1:
            # Not all cells are stimulated
            list_of_indices = [p.boundaries.find(tag) for tag in p.stimulus_tags]
            facets_gamma = np.array([], dtype=np.int32)
            for  l in list_of_indices:
                facets_gamma = np.concatenate((facets_gamma, l))
        elif len(p.gamma_tags) > 1:
            list_of_indices = [p.boundaries.find(tag) for tag in p.gamma_tags]
            facets_gamma = np.array([], dtype=np.int32)
            for l in list_of_indices:
                facets_gamma = np.concatenate((facets_gamma, l))
        else:
            facets_gamma = p.boundaries.values==p.gamma_tags[0]
        dofs_gamma   = dfx.fem.locate_dofs_topological(phi_M_space, p.mesh.topology.dim-1, facets_gamma) # The dofs of the gamma facets that are being stimulated 
        self.point_to_plot = dofs_gamma[0] # Choose one of the dofs as the point for plotting the membrane potential 
        
        self.v_t = []
        self.v_t.append(1000 * p.phi_M_prev.x.array[self.point_to_plot]) # Converted to mV
        self.out_v_string = self.out_file_prefix + 'v.png'

        if hasattr(p, 'n'):
            # Gating variables are a part of the problem
            self.n_t = []
            self.m_t = []
            self.h_t = []

            with p.n.x.petsc_vec.localForm() as local_n, \
                 p.m.x.petsc_vec.localForm() as local_m, \
                 p.h.x.petsc_vec.localForm() as local_h:
            
                self.n_t.append(local_n[self.point_to_plot])
                self.m_t.append(local_m[self.point_to_plot])
                self.h_t.append(local_h[self.point_to_plot])

            self.out_gate_string = self.out_file_prefix + 'gating.png'
            
    def save_png(self):
        """ Save data for the .png output of the membrane electric potential, and the gating variables if 
        these are a part of the problem. """

        p = self.problem

        self.v_t.append(1000 * p.phi_M_prev.x.array[self.point_to_plot]) # Converted to mV
        
        if hasattr(p, 'n'):
            with p.n.x.petsc_vec.localForm() as local_n, p.m.x.petsc_vec.localForm() as local_m, p.h.x.petsc_vec.localForm() as local_h:
                self.n_t.append(local_n[self.point_to_plot])
                self.m_t.append(local_m[self.point_to_plot])
                self.h_t.append(local_h[self.point_to_plot])

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
        time_steps = self.time_steps

        # Save plot of membrane potential
        plt.figure(0)        
        plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps + 1), self.v_t)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane potential (mV)')
        plt.savefig(self.out_v_string)

		# save plot of gating variables
        if hasattr(self.problem, 'n'):
            plt.figure(1)
            plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps + 1), self.n_t, label='n')
            plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps + 1), self.m_t, label='m')
            plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps + 1), self.h_t, label='h')
            plt.legend()
            plt.xlabel('Time (ms)')
            plt.savefig(self.out_gate_string)

        # Save iteration history
        if not self.direct_solver:
            plt.figure(2)
            plt.plot(self.iterations)
            plt.xlabel('Time step')
            plt.ylabel('Number of iterations')
            plt.savefig(self.out_file_prefix + 'iterations.png')

        # save runtime data
        plt.figure(3)
        plt.plot(self.assembly_time, label='assembly')
        plt.plot(self.solve_time, label='solve')
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Time (s)')
        plt.savefig(self.out_file_prefix + 'timings.png')

    def init_xdmf_savefile(self):
        """ Initialize .xdmf files for writing output. The mesh with meshtags is written to file,
        and an output file for the solution functions (ion concentrations and electric potentials)
        is initialized. """

        p = self.problem # For ease of notation

        # Write tag data
        filename = self.out_file_prefix + 'subdomains.xdmf'
        xdmf_file = dfx.io.XDMFFile(self.comm, filename, "w")
        xdmf_file.write_mesh(p.mesh)
        xdmf_file.write_meshtags(p.subdomains, p.mesh.geometry)
        xdmf_file.close()

        # Create solution file and write mesh to file
        filename = self.out_file_prefix + 'solution.xdmf'
        self.xdmf_file = dfx.io.XDMFFile(self.comm, filename, "w")
        self.xdmf_file.write_mesh(p.mesh)
        self.xdmf_file.write_meshtags(p.subdomains, p.mesh.geometry)
        self.output_filename = filename # Store the output filename for post-processing

        # Write solution functions to file
        for idx in range(p.N_ions+1):
            self.xdmf_file.write_function(p.u_p[0].sub(idx), float(p.t.value))
            self.xdmf_file.write_function(p.u_p[1].sub(idx), float(p.t.value))
        
        return

    def save_xdmf(self):
        """ Write solution functions (ion concentrations and electric potentials) to file. """

        for idx in range(self.problem.N_ions+1):
            self.xdmf_file.write_function(self.problem.u_p[0].sub(idx), float(self.problem.t.value))
            self.xdmf_file.write_function(self.problem.u_p[1].sub(idx), float(self.problem.t.value))

        return

    def close_xdmf(self):
        """ Close .xdmf files. """

        # Close the XDMF file
        self.xdmf_file.close()
        
        # Run XDMF parser to restructure the data for better visualization
        restructure_xdmf.run(self.output_filename)

        return
    
    # Default iterative solver parameters
    ksp_rtol           = 1e-6
    ksp_max_it         = 1000	
    ksp_type           = 'gmres'
    pc_type            = 'hypre'
    norm_type          = 'preconditioned'
    max_amg_iter       = 1
    use_P_mat          = True # use P as preconditioner?
    verbose            = False
    use_block_Jacobi   = True
    nonzero_init_guess = True

    # Default output save interval
    save_interval     = 1 # save every nth timestep

    # Iteration counter and time variables
    tot_its           = 0
    tot_assembly_time = 0
    tot_solver_time   = 0
