from mpi4py import MPI
import ufl
import numpy as np
import dolfinx as dfx
from CGx.KNPEMI.KNPEMIx_solver 	 	import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem 	import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley, ATPPump, NeuronalCotransporters

def main():
    """ Solve a KNP-EMI problem on a unit square with an iterative solver
        and assert that the computed L2 norms of the intra- and extracellular
        potentials are unchanged since the last test code update.

        Also assert that the average number of iterations used by the iterative solver is unchanged.

        The KNP-EMI problem is solved with the neuronal membrane mechanisms:
            - Hodgkin-Huxley model
            - ATP pump
            - Neuronal cotransporters
            
        Further details of the problem setup is defined in the configuration file:
        './src/CGx/KNPEMI/configs/tests/electric_potential_norms_iterative_solver.yaml'

    """
    # Create problem and initialize ionic models
    tags = {'intra': 1, 'extra': 2, 'boundary': 3, 'membrane': 4}
    problem_square = ProblemKNPEMI(
        config_file='./src/CGx/KNPEMI/configs/tests/electric_potential_norms_iterative_solver.yaml'
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
    saved_L2_phi_i = 3.510994056704844e-08
    saved_L2_phi_e = 6.369472309249516e-11

    print("\n---------------------------------------------------")
    print("Test: L2 norms of potentials using iterative solver")
    print("Saved phi_i:\t", saved_L2_phi_i)
    print("Computed phi_i:\t", computed_phi_i_L2_global)
    print("Saved phi_e:\t", saved_L2_phi_e)
    print("Computed phi_e:\t", computed_phi_e_L2_global)

    relative_error_phi_i = abs(computed_phi_i_L2_global - saved_L2_phi_i)/saved_L2_phi_i
    relative_error_phi_e = abs(computed_phi_e_L2_global - saved_L2_phi_e)/saved_L2_phi_e
    ksp_rtol = float(
        problem_square.solver_config['ksp_settings']['ksp_rtol'] # The relative tolerance used by the iterative solver
    )
    assert np.allclose(
            [relative_error_phi_i, relative_error_phi_e],
            [0.0, 0.0],
            atol=ksp_rtol*10 # The solver tolerance sets a limit on the achievable accuracy
        )
    
    # Check that the iterative solver uses the same number of
    # iterations as previously saved
    saved_iterations = 3.0
    current_iterations = sum(solver_square.iterations)/len(solver_square.iterations)
    print("\n---------------------------------------------------")
    print("Test: Number of iterations used by iterative solver")
    print("Saved number of iterations:\t", saved_iterations)
    print("Current number of iterations:\t", current_iterations)
    assert np.isclose(current_iterations, saved_iterations, atol=1e-8)

if __name__=='__main__':
    main()