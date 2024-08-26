import ufl
import argparse
import petsc4py.PETSc

from pathlib import Path
from CGx.utils.parsers import CustomParser
from CGx.EMI.EMIx_solver  import SolverEMI
from CGx.EMI.EMIx_problem import ProblemEMI
from CGx.EMI.EMIx_ionic_model import *

print = petsc4py.PETSc.Sys.Print

def main_yaml(yaml_file="config.yaml"):

    problem = ProblemEMI(yaml_file)

    HH = HH_model(problem, stim_fun=g_syn)
    ionic_models = [HH]

    problem.init_ionic_model(ionic_models)

    # Create solver and solve
    solver = SolverEMI(problem, save_xdmfs=True)
    solver.solve()

    tags = {'intra' : 1, 'extra' : 2, 'boundary' : 3, 'membrane' : 4}

    phi_i = solver.problem.wh[0]
    phi_e = solver.problem.wh[1]
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

	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--config", dest="config_file", default='./test_setup_config.yaml', type=Path, help="Configuration file")
	args = parser.parse_args(None)
	main_yaml(yaml_file=str(args.config_file))