""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import numpy as np

def evaluate_single_nuclide(results, n_points, cell, nuc, use_interpolation=True):
    """ Evaluates a single nuclide in a single cell from a results list.

    Parameters
    ----------
    results : list of results
        The results to extract data from.  Must be sorted and continuous.
    n_points : int
        Number of points, equally spaced, to evaluate on.
    cell : str
        Cell name to evaluate
    nuc : str
        Nuclide name to evaluate
    use_interpolation : bool
        Whether or not to use the algorithm-defined interpolation.
        n_points will be ignored.

    Returns
    -------
    time : numpy.array
        Time vector.
    concentration : numpy.array
        Total number of atoms in the cell.
    """

    if use_interpolation:
        time_final = results[-1].time[1]

        time = np.linspace(0, time_final, n_points)

        concentration = np.zeros(n_points)

        # Evaluate value in each region
        for res_i, result in enumerate(results):
            ind1 = np.argmax(time >= result.time[0])
            ind2 = np.argmax(time >= result.time[1])

            if res_i == len(results) - 1:
                ind2 = len(time)

            concentration[ind1:ind2] = result.evaluate(cell, nuc, time[ind1:ind2])
    else:
        n_points = len(results) + 1
        time = np.zeros(n_points)
        concentration = np.zeros(n_points)

        # Evaluate value in each region
        for i, result in enumerate(results):

            time[i] = result.time[0]
            time[i + 1] = result.time[1]

            poly = result[cell, nuc]

            concentration[i] = poly[0]
            concentration[i + 1] = np.sum(poly)

    return time, concentration

def evaluate_reaction_rate(results, cell, nuc, rxn):
    """ Evaluates a single nuclide reaction rate in a single cell from a results list.

    Parameters
    ----------
    results : list of Results
        The results to extract data from.  Must be sorted and continuous.
    cell : str
        Cell name to evaluate
    nuc : str
        Nuclide name to evaluate
    rxn : str
        Reaction rate to evaluate

    Returns
    -------
    time : numpy.array
        Time vector.
    rate : numpy.array
        Reaction rate.
    """

    n_points = len(results) + 1
    time = np.zeros(n_points)
    rate = np.zeros(n_points)

    ind_final = results[0].final_stage

    # Evaluate value in each region
    for i, result in enumerate(results):

        time[i] = result.time[0]
        time[i + 1] = result.time[1]

        poly = result[cell, nuc]

        rate[i] = result.rates[0][cell, nuc, rxn] * poly[0]
        rate[i + 1] = result.rates[ind_final][cell, nuc, rxn] * np.sum(poly)

    return time, rate

def evaluate_eigenvalue(results):
    """ Evaluates the eigenvalue from a results list.

    Parameters
    ----------
    results : list of Results
        The results to extract data from.  Must be sorted and continuous.

    Returns
    -------
    time : numpy.array
        Time vector.
    eigenvalue : numpy.array
        Eigenvalue.
    """

    n_points = len(results) + 1
    time = np.zeros(n_points)
    eigenvalue = np.zeros(n_points)

    ind_final = results[0].final_stage

    # Evaluate value in each region
    for i, result in enumerate(results):

        time[i] = result.time[0]
        time[i + 1] = result.time[1]

        eigenvalue[i] = result.k[0]
        eigenvalue[i + 1] = result.k[ind_final]

    return time, eigenvalue
