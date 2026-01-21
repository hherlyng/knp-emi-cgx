# knp-emi-cgx
Software for simulations of ionic electrodiffusion based on the KNP-EMI equations, using the FEniCS Python interface DOLFINx.

# Installation
## Conda
Clone this repository and navigate into its root directory `knp-emi-cgx`. <br>
Set up a conda environment by using the configuration file `environment.yml` located in the root directory. The environment is automatically named CGx. Creating the `conda` environment and activating it is done by running <br> 
```
conda env create -f environment.yml
conda activate CGx
```

Whilst still in the root directory, install the `CGx` Python module by running

```
pip install -e . --no-build-isolation
```

# Testing the setup on a unit square mesh
To test the setup by solving the KNP-EMI problem on a unit square, start by generating a mesh
with the command
```
python3 ./src/CGx/utils/generate_square_mesh.py -N 32 -o './input/geometries/'
```
This generates a 32x32 unit square mesh in the directory `'./input/geometries/'`.
The mesh consists of triangles and the option `-N` specifies the number
of elements in the $x$ and $y$ directions.
Test your setup by running 
```
python3 ./tests/KNPEMI/electric_potential_norms_direct_solver.py
```
This solves a KNP-EMI problem on the unit square mesh with a direct solver.
To test solving the same problem with an iterative solver, run
```
python3 ./tests/KNPEMI/electric_potential_norms_iterative_solver.py
```

A general `main` file is found in the `src/CGx/KNPEMI` directory.
This file is used to run simulations using configuration files
that are located in `src/CGx/KNPEMI/configs`, using the command
```
python main.py --config your_config_file.yaml
```

# Citation and example use cases
If you use this code, please cite the following paper, which is where
the numerical method was developed:
```
@article{benedusi2024scalable,
  author = {Benedusi, Pietro and Ellingsrud, Ada Johanne and Herlyng, Halvor and Rognes, Marie E.},
  title = {Scalable Approximation and Solvers for Ionic Electrodiffusion in Cellular Geometries},
  journal = {SIAM Journal on Scientific Computing},
  volume = {46},
  number = {5},
  pages = {B725-B751},
  year = {2024},
  doi = {10.1137/24M1644717}
}
```
Note that, in the above article, the legacy FEniCS implementation 
located in [this repository](https://github.com/pietrobe/EMIx/) is used.

For example use cases of the software in the current repository,
which implements the methodology in the above article in FEniCSx,
see the following paper:
```
@misc{herlyng2025modelingsimulationelectrodiffusiondense,
  title={Modeling and simulation of electrodiffusion in dense reconstructions of cerebral tissue}, 
  author={Halvor Herlyng and Marius Causemann and Gaute T. Einevoll and Ada J. Ellingsrud and Geir Halnes and Marie E. Rognes},
  year={2025},
  eprint={2512.03224},
  archivePrefix={arXiv},
  primaryClass={physics.med-ph},
  url={https://arxiv.org/abs/2512.03224}, 
}
```
