from mpi4py import MPI
from CGx.KNPEMI.KNPEMIx_problem import ProblemKNPEMI

import ufl
import numpy   as np
import dolfinx as dfx

def create_flux_forms(problem: ProblemKNPEMI) -> list[dfx.fem.Form]:
    """Compute the molar fluxes for each ion species across the cellular membrane.
       The fluxes have units of mol/s.

    Parameters
    ----------
    problem : ProblemKNPEMI
        The KNPEMI problem instance.

    Returns
    -------
    list[dfx.fem.Form]
        A list of compiled forms for the molar fluxes (mol/s)
        for each ion species across the cellular membrane.
    """
    flux_forms = []
    dS: ufl.Measure = problem.dS
    wh: list[dfx.fem.Function] = problem.wh
    N_ions: int = problem.N_ions
    ions: list[dict] = problem.ion_list
    gamma_tag: int = problem.membrane_data_tag
    n = ufl.FacetNormal(problem.mesh)
    i_res = "+"  # Intracellular facet restriction
    e_res = "-"  # Extracellular facet restriction

    # Extract constants
    psi: dfx.fem.Constant = problem.psi

    if not problem.stimulus_region:
        mask = 1.0
    else:
        # Create mask so that stimulus is zero outside of 
        # a subregion, but active within the subregion
        x = ufl.SpatialCoordinate(problem.mesh)
        coord = x[problem.stimulus_region_direction]
        coord_min: float = problem.stimulus_region_range[0]
        coord_max: float = problem.stimulus_region_range[1]
        mask = ufl.conditional(
                    ufl.And(
                        ufl.gt(coord, coord_min), 
                        ufl.lt(coord, coord_max)
                        ), 
                    1.0, 0.0
                )

    for space_idx, res in enumerate([i_res, e_res]):  # 0: ICS, 1: ECS
        # Extract electric potential function
        phi: dfx.fem.Function = wh[space_idx][N_ions]  # Electric potential

        for ion_idx, ion in enumerate(ions):
            # Extract concentration and potential functions
            c: dfx.fem.Function = wh[space_idx][ion_idx] 

            # Get name, diffusion coefficient and valence for the ion
            D_ion: dfx.fem.Constant = ion['Di'] # Diffusion coefficient, intra- and extracellular assumed equal
            z_ion: dfx.fem.Constant = ion['z'] # Valence

            # Define the flux expression
            grad_c   = ufl.grad(c)   # Concentration gradient
            grad_phi = ufl.grad(phi) # Electric potential gradient
            flux_expr: ufl.Coefficient = -D_ion * (grad_c + (z_ion / psi) * c * grad_phi) # Total flux

            # Compute the flux
            flux: ufl.Form = mask * ufl.dot(flux_expr(res), n(res)) * dS(gamma_tag)
            flux_form:  float = dfx.fem.form(flux)
            flux_forms.append(flux_form) # Store the flux form

    return flux_forms

def compute_fluxes(flux_forms: list[dfx.fem.Form], comm: MPI.Comm) -> np.ndarray[float]:
    """Compute the L2 norms of the fluxes for each ion species in intra- and extracellular spaces.

    Parameters
    ----------
    flux_norms : list[dfx.fem.Form]
        The flux norms to compute.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    np.ndarray[float]
        The computed L2 norms of the fluxes.
    """
    return np.array(
                [comm.allreduce(
                    dfx.fem.assemble_scalar(
                        dfx.fem.form(norm)
                    ),
                    op=MPI.SUM
                )
                for norm in flux_forms]
            )