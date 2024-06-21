import argparse
import petsc4py.PETSc

from pathlib import Path
from CGx.KNPEMI.KNPEMIx_ionic_model import *
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_solver import SolverKNPEMI
from CGx.KNPEMI.parsers import CustomParser

print = petsc4py.PETSc.Sys.Print

def main(argv=None):
	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--input", dest="input_file", default='geometries/square32.xdmf', type=Path, help="Input file")

	marker_opts = parser.add_argument_group("Domain markers", "Tags for the different domains")
	marker_opts.add_argument("-i", "--intra" ,default=1, type=int, help="Intracellular tag")
	marker_opts.add_argument("-e", "--extra" ,default=2, type=int, help="Extracellular tag")
	marker_opts.add_argument("-b", "--boundary" ,default=3, type=int, help="Boundary tag")
	marker_opts.add_argument("-m", "--membrane" ,default=4, type=int, help="Membrane tag")

	temporal_opts = parser.add_argument_group("Temporal options", "Options for the temporal discretization")
	temporal_opts.add_argument("--dt", default=5e-5, type=float, help="Time step")
	temporal_opts.add_argument("--time_steps", default=50, type=int, help="Number of time steps")

	solver_opts = parser.add_argument_group("Solver options", "Options for the linear solver")
	solver_opts.add_argument("-d", "--direct", default=1, type=int, help="Use direct solver")

	args = parser.parse_args(argv)
	tags = {'intra' : args.intra, 'extra' : args.extra, 'boundary' : args.boundary, 'membrane' : args.membrane}

	# Create problem
	problem = ProblemKNPEMI(args.input_file, tags, args.dt)
	# Set ionic models
	HH = HH_model(problem)
	ionic_models = [HH]

	problem.init_ionic_model(ionic_models)

	# Create solver and solve
	solver = SolverKNPEMI(problem, args.time_steps, args.direct)
	solver.solve()

	phi_i = solver.problem.wh[0].sub(problem.N_ions)
	phi_e = solver.problem.wh[1].sub(problem.N_ions)
	dx = solver.problem.dx

	phi_i_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * dx(tags['intra'])))
	phi_i_L2_global = solver.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
	phi_i_L2_global = np.sqrt(phi_i_L2_global)
	
	phi_e_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * dx(tags['extra'])))
	phi_e_L2_global = solver.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
	phi_e_L2_global = np.sqrt(phi_e_L2_global)
	
	print(f"L2 norm phi_i = {phi_i_L2_global}")
	print(f"L2 norm phi_e = {phi_e_L2_global}")

def main_yaml(yaml_file="config.yaml"):
	
	problem = ProblemKNPEMI(yaml_file)

	HH = HH_model(problem)
	ionic_models = [HH]

	problem.init_ionic_model(ionic_models)

	# Create solver and solve
	solver = SolverKNPEMI(problem)
	solver.solve()

	tags = {'intra' : 1, 'extra' : 2, 'boundary' : 3, 'membrane' : 4}

	phi_i = solver.problem.wh[0].sub(problem.N_ions)
	phi_e = solver.problem.wh[1].sub(problem.N_ions)
	dx = solver.problem.dx

	phi_i_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_i, phi_i) * dx(tags['intra'])))
	phi_i_L2_global = solver.comm.allreduce(phi_i_L2_local, op=MPI.SUM)
	phi_i_L2_global = np.sqrt(phi_i_L2_global)
	
	phi_e_L2_local = dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(phi_e, phi_e) * dx(tags['extra'])))
	phi_e_L2_global = solver.comm.allreduce(phi_e_L2_local, op=MPI.SUM)
	phi_e_L2_global = np.sqrt(phi_e_L2_global)
	
	print(f"L2 norm phi_i = {phi_i_L2_global}")
	print(f"L2 norm phi_e = {phi_e_L2_global}")

if __name__=='__main__':

	main_yaml()


