# knp-emi-cgx
Software for simulations of ionic electrodiffusion based on the KNP-EMI equations, using the FEniCS Python interface DOLFINx.

# Installation
## Conda
Clone this repository and navigate into its root directory `knp-emi-cgx`. \\ 
Set up a conda environment by running \\ `conda env create -f environment.yml`
Activate the environment with `conda activate CGx`.
Install the `CGx` Python module by running `pip install -e .` (when in the root directory).

# Testing the setup
The setup can be tested by solving the KNP-EMI problem on a 32x32 unit square by doing:
Navigate into the KNP-EMI directory: `cd src/CGx/KNPEMI`
Run `python main.py --config test_setup_config.yaml`