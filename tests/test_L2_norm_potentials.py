from CGx.KNPEMI.KNPEMIx_solver 	 	import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem 	import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import *

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
		problem_square = ProblemKNPEMI(input_file='./square32.xdmf', tags=tags, dt=5e-5)
		HH = HH_model(problem_square)
		ionic_models = [HH]
		problem_square.init_ionic_model(ionic_models)

		# Create solver
		solver_square = SolverKNPEMI(problem_square, time_steps=10, direct=True)
		solver_square.solve()

		# Extract the solutions of the potentials
		phi_i = solver_square.problem.wh[0].sub(solver_square.problem.N_ions)
		phi_e = solver_square.problem.wh[1].sub(solver_square.problem.N_ions)

		# Calculate the L2 norms
		phi_i_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * problem_square.dx(tags['intra'])))
		phi_i_L2_global = solver_square.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
		phi_i_L2_global = np.sqrt(phi_i_L2_global)
		
		phi_e_L2_local  = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * problem_square.dx(tags['extra'])))
		phi_e_L2_global = solver_square.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
		phi_e_L2_global = np.sqrt(phi_e_L2_global)

		# Calculate the percentage errors and assert that they're smaller than 1%, when
		# comparing the calculated L2 norms of the potentials with previously saved values.
		saved_L2_phi_i = 0.015551359556518578 #mesh_conversion_factor=1e-6 : 1.5403605575577102e-08
		saved_L2_phi_e = 0.009009757241662388 #mesh_conversion_factor=1e-6 : 8.924158982124452e-09

		percentage_error_i = abs(phi_i_L2_global - saved_L2_phi_i)/saved_L2_phi_i*100
		percentage_error_e = abs(phi_e_L2_global - saved_L2_phi_e)/saved_L2_phi_e*100
		
		assert all([percentage_error_i < 1, percentage_error_e < 1])

if __name__=='__main__':
    test_L2_norm_of_potentials()