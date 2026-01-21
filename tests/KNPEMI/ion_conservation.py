from mpi4py import MPI
import ufl
import numpy as np
import dolfinx as dfx
from CGx.KNPEMI.KNPEMIx_solver 	 	import SolverKNPEMI
from CGx.KNPEMI.KNPEMIx_problem 	import ProblemKNPEMI
from CGx.KNPEMI.KNPEMIx_ionic_model import HodgkinHuxley, ATPPump, NeuronalCotransporters

def main():
    print("Hello, World!")

if __name__=='__main__':
    main()