from sys                 import argv
from mpi4py              import MPI
from KNPEMIx_solver      import SolverKNPEMI
from KNPEMIx_problem     import ProblemKNPEMI
from KNPEMIx_ionic_model import *

if __name__=='__main__':

	comm = MPI.COMM_WORLD # MPI communicator
	
	# global time step (s) and number of timesteps
	dt = 5e-5
	time_steps = 50

	# square
	N = int(argv[1]) # Number of mesh cells
	input_file = f'geometries/square{N}.xdmf'	
	tags = {'intra' : 1, 'extra' : 2, 'boundary' : 3, 'membrane' : 4}

	# # astrocyte
	# input_file = 'geometries/astrocyte_mesh_full.xdmf'
	# tags = {'intra' : 3, 'extra' : 1, 'boundary' : 4, 'membrane' : 5}

	# # dendrite
	# input_file = '../../../data/dendrite/dfx_mesh.xdmf'
	# tags = {'intra' : (2, 3, 4) , 'extra' : 1, 'boundary' : 1, 'membrane' : (2, 3, 4)}

	# GC
	# input_file = '../../../data/GC/'

	# Create problem
	problem = ProblemKNPEMI(input_file, tags, dt)

	# Set ionic models
	HH = HH_model(problem)
	ionic_models = [HH]

	problem.init_ionic_model(ionic_models)

	# Create solver and solve
	solver = SolverKNPEMI(problem, time_steps)
	solver.solve()