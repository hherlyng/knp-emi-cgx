import petsc4py
import argparse

from pathlib import Path
from CGx.utils.parsers import CustomParser
from CGx.KNPEMI.KNPEMIx_solver  import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import *

pprint = print
print = petsc4py.PETSc.Sys.Print

def main_yaml(yaml_file: str="config.yaml"):
    """ Main for running scripts with a yaml/yml configuration file.

    Parameters
    ----------
    yaml_file : str, optional
        The path to the yaml configuration file, by default "config.yaml"
    ksp_view : bool, optional
        Iterative solver option used to view information of KSP object, by default False
    """

    problem = ProblemKNPEMI(yaml_file)
    passive_model = PassiveModel(problem)
    problem.set_initial_conditions()
    problem.init_ionic_model([passive_model])

    tic = time.perf_counter()
    problem.setup_variational_form()
    var_form_setup_time = time.perf_counter()-tic
    var_form_setup_time_max = problem.comm.allreduce(var_form_setup_time, op=MPI.MAX)
    print(f"Variational form setup in {var_form_setup_time_max:0.4f} seconds")

    # Create solver and solve
    solver = SolverKNPEMI(problem,
                            view_input=False,
                            save_xdmfs=True,
                            use_direct_solver=True,
                            save_pngs=True,
                            save_cpoints=False
                        )
    solver.solve()

if __name__=='__main__':
	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--config", dest="config_file", type=Path, help="Configuration file")
	args = parser.parse_args(None)
	main_yaml(yaml_file=str(args.config_file))