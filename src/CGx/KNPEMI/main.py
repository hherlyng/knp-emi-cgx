import argparse
import petsc4py.PETSc

from pathlib import Path
from CGx.utils.parsers import CustomParser
from CGx.KNPEMI.KNPEMIx_solver  import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import *

pprint = print
print = petsc4py.PETSc.Sys.Print

def main_yaml(yaml_file="config.yaml", view_input=None):
	
	problem = ProblemKNPEMI(yaml_file)

	# Set ionic models
	HH = HodgkinHuxley(problem, tags=problem.neuron_tags, stimulus=False)
	ATP = ATPPump(problem, tags=problem.neuron_tags)
	NeuronalCT = NeuronalCotransporters(problem, tags=problem.neuron_tags)
	KirNa = KirNaKPumpModel(problem, tags=problem.glia_tags)
	GlialCT = GlialCotransporters(problem, tags=problem.glia_tags)

	ionic_models = [HH, ATP, NeuronalCT]#, KirNa, GlialCT]

	problem.init_ionic_model(ionic_models)

	# Create solver and solve
	solver = SolverKNPEMI(problem,
						  view_input=view_input,
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
	parser.add_argument("--view", dest="view_input", type=bool)
	args = parser.parse_args(None)
	main_yaml(yaml_file=str(args.config_file), view_input=bool(args.view_input))