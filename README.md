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

# Testing the setup
The setup can be tested by solving the KNP-EMI problem on a 32x32 unit square by navigating to the KNP-EMI directory and running the main file:

```
cd src/CGx/KNPEMI
python main.py --config test_setup_config.yaml
```
