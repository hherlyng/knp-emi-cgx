import argparse
from pathlib import Path

from KNPEMIx_ionic_model import *
from KNPEMIx_problem import ProblemKNPEMI
from KNPEMIx_solver import SolverKNPEMI
from parsers import CustomParser


def main(argv=None):
	parser = argparse.ArgumentParser(formatter_class=CustomParser)
	parser.add_argument("--input", dest="input_file", default='geometries/square.xdmf', type=Path, help="Input file")
	marker_opts = parser.add_argument_group("Domain markers", "Tags for the different domains")
	marker_opts.add_argument("-i", "--intra" ,default=1, type=int, help="Intracellular tag")
	marker_opts.add_argument("-e", "--extra" ,default=2, type=int, help="Extracellular tag")
	marker_opts.add_argument("-b", "--boundary" ,default=3, type=int, help="Boundary tag")
	marker_opts.add_argument("-m", "--membrane" ,default=4, type=int, help="Membrane tag")
	temporal_opts = parser.add_argument_group("Temporal options", "Options for the temporal discretization")
	temporal_opts.add_argument("--dt", default=5e-5, type=float, help="Time step")
	temporal_opts.add_argument("--time_steps", default=50, type=int, help="Number of time steps")
	args = parser.parse_args(argv)
	tags = {'intra' : args.intra, 'extra' : args.extra, 'boundary' : args.boundary, 'membrane' : args.membrane}

	# Create problem
	problem = ProblemKNPEMI(args.input_file, tags, args.dt)
	# Set ionic models
	HH = HH_model(problem)
	ionic_models = [HH]

	problem.init_ionic_model(ionic_models)

	# Create solver and solve
	solver = SolverKNPEMI(problem, args.time_steps)
	solver.solve()

if __name__=='__main__':

	main()
	# # astrocyte
	# input_file = 'geometries/astrocyte_mesh_full.xdmf'
	# tags = {'intra' : 3, 'extra' : 1, 'boundary' : 4, 'membrane' : 5}

	# # dendrite
	# input_file = '../../../data/dendrite/dfx_mesh.xdmf'
	# tags = {'intra' : (2, 3, 4) , 'extra' : 1, 'boundary' : 1, 'membrane' : (2, 3, 4)}

	# GC
	# input_file = '../../../data/GC/'


