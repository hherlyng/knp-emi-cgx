import petsc4py
import argparse

from pathlib import Path
from CGx.utils.parsers import CustomParser
from CGx.KNPEMI.KNPEMIx_solver  import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import *

print = petsc4py.PETSc.Sys.Print

def main_yaml(yaml_file: str="config.yaml", view_ksp: bool=False):
	""" Main for running scripts with a yaml/yml configuration file.

	Parameters
	----------
	yaml_file : str, optional
		The path to the yaml configuration file, by default "config.yaml"
	ksp_view : bool, optional
		Iterative solver option used to view information of KSP object, by default False
	"""
	
	# Instantiate problem using config file
	problem = ProblemKNPEMI(yaml_file)

	# Set ionic models
	if "square_config" in yaml_file:
		HH = HodgkinHuxley(problem)
		ATP = ATPPump(problem)
		NeuronalCT = NeuronalCotransporters(problem)
		ionic_models = [NeuronalCT, HH, ATP]
	else:
		HH = HodgkinHuxley(problem, tags=problem.neuron_tags)
		ATP = ATPPump(problem, tags=problem.neuron_tags)
		NeuronalCT = NeuronalCotransporters(problem, tags=problem.neuron_tags)
		KirNa = KirNaKPumpModel(problem, tags=problem.glia_tags)
		GlialCT = GlialCotransporters(problem, tags=problem.glia_tags)

		ionic_models = [HH, ATP, NeuronalCT, GlialCT, KirNa]
		
	problem.set_initial_conditions()
	problem.init_ionic_models(ionic_models)

	# Time the variational form setup
	tic = time.perf_counter()
	problem.setup_variational_form()
	var_form_setup_time = time.perf_counter()-tic
	var_form_setup_time_max = problem.comm.allreduce(var_form_setup_time, op=MPI.MAX)
	print(f"Variational form setup in {var_form_setup_time_max:0.4f} seconds")
	
	# Create solver and solve
	# Get solver config
	solver_config = problem.solver_config
	solver = SolverKNPEMI(problem,
						  view_input=view_ksp,
						  save_xdmfs=solver_config['output']['save_xdmf'],
						  use_direct_solver=solver_config['direct'],
						  save_pngs=solver_config['output']['save_pngs'],
						  save_cpoints=solver_config['output']['save_cpoints'],
						  save_dat=solver_config['output']['save_dat']
						)
	solver.solve()

	# Calculate the L2 norm of the intra- and extracellular potentials
	phi_i: dfx.fem.Function = solver.problem.wh[0][problem.N_ions]
	phi_e: dfx.fem.Function = solver.problem.wh[1][problem.N_ions]
	dx: ufl.Measure = solver.problem.dx

	phi_i_L2_local: float = dfx.fem.assemble_scalar(
								dfx.fem.form(
									ufl.inner(phi_i, phi_i) * dx(problem.intra_tags)
								)
							)
	phi_i_L2_global = solver.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
	phi_i_L2_global = np.sqrt(phi_i_L2_global)

	phi_e_L2_local: float = dfx.fem.assemble_scalar(
								dfx.fem.form(
									ufl.inner(phi_e, phi_e) * dx(problem.extra_tag)
								)
							)
	phi_e_L2_global = solver.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
	phi_e_L2_global = np.sqrt(phi_e_L2_global)
	
	print(f"L2 norm phi_i = {phi_i_L2_global}")
	print(f"L2 norm phi_e = {phi_e_L2_global}")

if __name__=='__main__':

	# Parse command line arguments
	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--config", dest="config_file", default='./test_setup_config.yaml', type=Path, help="Configuration file")
	parser.add_argument("--view", dest="view_ksp", default=False, type=bool, help="Verbose KSP object log")
	
	args = parser.parse_args(None)
	
	# Start script timer
	script_time_tic = time.perf_counter()

	# Run main
	main_yaml(yaml_file=str(args.config_file), view_ksp=args.view_ksp)
	
	# End script timer and print
	script_time_toc = time.perf_counter()
	script_time = MPI.COMM_WORLD.allreduce(script_time_toc - script_time_tic, op=MPI.MAX)
	print(f"Total script time: {script_time:0.4f} seconds")