import argparse
import petsc4py.PETSc

from pathlib import Path
from CGx.utils.parsers import CustomParser
from CGx.KNPEMI.KNPEMIx_solver  import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import *

pprint = print
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
	
	problem = ProblemKNPEMI(yaml_file)

	if problem.MMS_test:
		ionic_models = [PassiveModel(problem, tags=(1, 2, 3, 4))]
	else:
		# Set ionic models
		HH = HodgkinHuxley(problem, tags=problem.gamma_tags)
		# HH = HodgkinHuxley(problem, tags=problem.neuron_tags)
		# ATP = ATPPump(problem, tags=problem.neuron_tags)
		# NeuronalCT = NeuronalCotransporters(problem, tags=problem.neuron_tags)
		# KirNa = KirNaKPumpModel(problem, tags=problem.glia_tags)
		# GlialCT = GlialCotransporters(problem, tags=problem.glia_tags)

		ionic_models = [HH]#, ATP, NeuronalCT, GlialCT, KirNa]

	problem.init_ionic_model(ionic_models)
	problem.set_initial_conditions()
	problem.setup_variational_form()

	# Create solver and solve
	solver = SolverKNPEMI(problem,
						  view_input=view_ksp,
						  save_xdmfs=True,
						  use_direct_solver=False,
						  save_pngs=True,
						  save_cpoints=True)
	solver.solve()

	phi_i = solver.problem.wh[0].sub(problem.N_ions)
	phi_e = solver.problem.wh[1].sub(problem.N_ions)
	dx = solver.problem.dx

	phi_i_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * dx(problem.intra_tags)))
	phi_i_L2_global = solver.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
	phi_i_L2_global = np.sqrt(phi_i_L2_global)
	
	phi_e_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * dx(problem.extra_tag)))
	phi_e_L2_global = solver.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
	phi_e_L2_global = np.sqrt(phi_e_L2_global)
	
	print(f"L2 norm phi_i = {phi_i_L2_global}")
	print(f"L2 norm phi_e = {phi_e_L2_global}")

if __name__=='__main__':

	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--config", dest="config_file", default='./test_setup_config.yaml', type=Path, help="Configuration file")
	parser.add_argument("--view", dest="view_ksp", default=False, type=bool, help="Verbose KSP object log")
	args = parser.parse_args(None)
	main_yaml(yaml_file=str(args.config_file), view_ksp=args.view_ksp)