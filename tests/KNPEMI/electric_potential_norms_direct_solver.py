from mpi4py import MPI
import ufl
import numpy as np
import dolfinx as dfx
from CGx.KNPEMI.KNPEMIx_solver 	 	import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem 	import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley, ATPPump, NeuronalCotransporters

def main():
	""" Solve a KNP-EMI problem with a direct solver on a unit square and assert
      	that the computed L2 norms of the intra- and extracellular potentials
        are unchanged since the last test code update.
        
        The KNP-EMI problem is solved with the neuronal membrane mechanisms:
            - Hodgkin-Huxley model
            - ATP pump
            - Neuronal cotransporters
            
        Further details of the problem setup is defined in the configuration file:
        './src/CGx/KNPEMI/configs/tests/electric_potential_norms_direct_solver.yaml'
	"""
	# Create problem and initialize ionic models
	tags = {'intra': 1, 'extra': 2, 'boundary': 3, 'membrane': 4}
	problem_square = ProblemKNPEMI(
		config_file='./src/CGx/KNPEMI/configs/tests/electric_potential_norms_direct_solver.yaml'
	)
	HH = HodgkinHuxley(problem_square)
	ATP = ATPPump(problem_square)
	NeuronalCT = NeuronalCotransporters(problem_square)
	ionic_models = [NeuronalCT, HH, ATP]
	problem_square.set_initial_conditions()
	problem_square.init_ionic_models(ionic_models)
	problem_square.setup_variational_form()

	# Create solver
	problem_square.solver_config['view_ksp'] = False # Set view_ksp option to False, not needed for test
	solver_square = SolverKNPEMI(problem_square, solver_config=problem_square.solver_config)
	solver_square.solve()

	# Extract the solutions of the potentials
	phi_i = solver_square.problem.wh[0][solver_square.problem.N_ions]
	phi_e = solver_square.problem.wh[1][solver_square.problem.N_ions]

	# Calculate the L2 norms
	phi_i_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * problem_square.dx(tags['intra'])))
	phi_i_L2_global = solver_square.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
	computed_phi_i_L2_global = np.sqrt(phi_i_L2_global)

	phi_e_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * problem_square.dx(tags['extra'])))
	phi_e_L2_global = solver_square.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
	computed_phi_e_L2_global = np.sqrt(phi_e_L2_global)

	# Calculate the percentage errors and assert that they're smaller than 1%, when
	# comparing the computed L2 norms of the potentials with previously saved values.
	saved_L2_phi_i = 2.6337161145147203e-08
	saved_L2_phi_e = 1.5258564901943312e-08

	print("\n------------------------------------------------")
	print("Test: L2 norms of potentials using direct solver")
	print("Saved phi_i:\t", saved_L2_phi_i)
	print("Computed phi_i:\t", computed_phi_i_L2_global)
	print("Saved phi_e:\t", saved_L2_phi_e)
	print("Computed phi_e:\t", computed_phi_e_L2_global)

	percentage_error_i = abs(computed_phi_i_L2_global - saved_L2_phi_i)/saved_L2_phi_i*100
	percentage_error_e = abs(computed_phi_e_L2_global - saved_L2_phi_e)/saved_L2_phi_e*100
	
	assert np.allclose([percentage_error_i, percentage_error_e], [0.0, 0.0], atol=1e-8)

if __name__=='__main__':
    main()