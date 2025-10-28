from mpi4py import MPI
import ufl
import numpy as np
import dolfinx as dfx
from CGx.KNPEMI.KNPEMIx_solver 	 	import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem 	import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley

def test_L2_norm_of_potentials():
		""" Solve the KNP-EMI problem on a unit square and assert that the calculated
			L2 norms of the intra- and extracellular potentials are unchanged since the last
			code update.

			Comparison with L2 norms calculated using
				- Unit square with N=32
				- dt = 5e-5
				- time_steps = 10
		"""
		# Create problem and initialize ionic models
		tags = {'intra': 1, 'extra': 2, 'boundary': 3, 'membrane': 4}
		problem_square = ProblemKNPEMI(config_file='./tests/test_config.yml')
		HH = HodgkinHuxley(problem_square)
		problem_square.init_ionic_models([HH])
		problem_square.set_initial_conditions()
		problem_square.setup_variational_form()

		# Create solver
		solver_square = SolverKNPEMI(problem_square, view_input=False)
		solver_square.solve()

		# Extract the solutions of the potentials
		phi_i = solver_square.problem.wh[0][solver_square.problem.N_ions]
		phi_e = solver_square.problem.wh[1][solver_square.problem.N_ions]

		# Calculate the L2 norms
		phi_i_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * problem_square.dx(tags['intra'])))
		phi_i_L2_global = solver_square.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
		phi_i_L2_global = np.sqrt(phi_i_L2_global)

		phi_e_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * problem_square.dx(tags['extra'])))
		phi_e_L2_global = solver_square.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
		phi_e_L2_global = np.sqrt(phi_e_L2_global)

		# Calculate the percentage errors and assert that they're smaller than 1%, when
		# comparing the calculated L2 norms of the potentials with previously saved values.
		saved_L2_phi_i = 1.9331311559715952e-11
		saved_L2_phi_e = 0.020961705521519262

		print("Saved phi_i:\t", saved_L2_phi_i)
		print("Calculated phi_i:\t", phi_i_L2_global)
		print("Saved phi_e:\t", saved_L2_phi_e)
		print("Calculated phi_e:\t", phi_e_L2_global)

		percentage_error_i = abs(phi_i_L2_global - saved_L2_phi_i)/saved_L2_phi_i*100
		percentage_error_e = abs(phi_e_L2_global - saved_L2_phi_e)/saved_L2_phi_e*100

		assert all([percentage_error_i < 1, percentage_error_e < 1])

if __name__=='__main__':
    test_L2_norm_of_potentials()