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
pip install -e .
```

# Testing the setup on a unit square mesh
To test the setup by solving the KNP-EMI problem on a unit square, start by generating a mesh. Navigate to the KNP-EMI directory
```
cd src/CGx/KNPEMI
```
and run
```
python ../utils/generate_square_mesh.py
```
to generate a 32x32 unit square mesh. The script `generate_square_mesh.py` has an option -N which can be provided to generate an NxN unit square mesh.
To run a simulation with a test setup config, run
```
python main.py --config test_setup_config.yaml
```

# Citation
If you use this code, please cite the following paper
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
