""" The CE/CM integrator.

Implements the CE/CM Predictor-Corrector algorithm.

This algorithm is mathematically defined as:

.. math:
    y' = A(y, t) y(t)
    A_p = A(y_n, t_n)
    y_m = expm(A_p h/2) y_n
    A_c = A(y_m, t_n + h/2)
    y_{n+1} = expm(A_c h) y_n
"""

import copy
import os
import time

from mpi4py import MPI

from .cram import CRAM48
from .save_results import save_results

def cecm(operator, print_out=True):
    """ Performs integration of an operator using the CECM pc algorithm.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    print_out : bool, optional
        Whether or not to print out time.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    n_mats = len(vec)

    t = 0.0

    for i, dt in enumerate(operator.settings.dt_vec):
        # Create vectors
        x = [copy.copy(vec)]
        seeds = []
        eigvls = []
        rates_array = []

        eigvl, rates, seed = operator.eval(x[0])

        eigvls.append(eigvl)
        seeds.append(seed)
        rates_array.append(rates)

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f = operator.form_matrix(rates_array[0], mat)

            x_new = CRAM48(f, x[0][mat], dt/2)

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        x.append(x_result)

        eigvl, rates, seed = operator.eval(x[1])

        eigvls.append(eigvl)
        seeds.append(seed)
        rates_array.append(rates)

        x_result = []

        t_start = time.time()
        for mat in range(n_mats):
            # Form matrix
            f = operator.form_matrix(rates_array[1], mat)

            x_new = CRAM48(f, x[0][mat], dt)

            x_result.append(x_new)

        t_end = time.time()
        if MPI.COMM_WORLD.rank == 0:
            if print_out:
                print("Time to matexp: ", t_end - t_start)

        # Create results, write to disk
        save_results(operator, x, rates_array, eigvls, seeds, [t, t + dt], i)

        t += dt
        vec = copy.deepcopy(x_result)

    # Perform one last simulation
    x = [copy.copy(vec)]
    seeds = []
    eigvls = []
    rates_array = []
    eigvl, rates, seed = operator.eval(x[0])

    eigvls.append(eigvl)
    seeds.append(seed)
    rates_array.append(rates)

    # Create results, write to disk
    save_results(operator, x, rates_array, eigvls, seeds, [t, t],
                 len(operator.settings.dt_vec))

    # Return to origin
    os.chdir(dir_home)
