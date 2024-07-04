import petsc4py.PETSc

from CGx.EMI.EMIx_solver      import SolverEMI
from CGx.EMI.EMIx_problem     import ProblemEMI
from CGx.EMI.EMIx_ionic_model import *

print = petsc4py.PETSc.Sys.Print

def main():

    problem = ProblemEMI(config_file="mms_config.yml")

    passive_model = Passive_model(problem)
    ionic_models = [passive_model]

    problem.add_ionic_model(ionic_models, problem.gamma_tags, stim_fun=g_syn)
    problem.init_ionic_model(ionic_models)

    # Create solver and solve
    solver = SolverEMI(problem, save_xdmfs=True)
    solver.solve()

if __name__=='__main__':
	main()