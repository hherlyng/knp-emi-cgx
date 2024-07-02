# knp-emi-cgx
Software for simulations of ionic electrodiffusion based on the KNP-EMI equations, using the FEniCS Python interface DOLFINx.

# Installation
## Conda
Clone this repository and navigate into its root directory `knp-emi-cgx`. <br>
Set up a conda environment and activate it (will automatically be named CGx) by running <br> 
```conda env create -f environment.yml
conda activate CGx``` <br>
Install the `CGx` Python module by (when in the root directory) running <br>
`pip install -e .` 

# Testing the setup
The setup can be tested by solving the KNP-EMI problem on a 32x32 unit square by navigating to the KNP-EMI directory and running the main file: <br>
`cd src/CGx/KNPEMI` <br>
`python main.py --config test_setup_config.yaml`