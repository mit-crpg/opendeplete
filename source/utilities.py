""" The utilities module.

Contains functions that can be used to post-process objects that come out of
the results module.
"""

import numpy as np
import os
import fnmatch
import pickle


def get_eigval(directory):
    """ Get eigenvalues as a function of time.

    Parameters
    ----------
    directory : str
        Directory to read results from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : np.array
        Eigenvalue for each step.
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = np.zeros(count)
    time = np.zeros(count)

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            handle = open(directory + '/' + file, 'rb')
            result = pickle.load(handle)
            handle.close()

            # Extract results
            val[ind] = result.k
            time[ind] = result.time
    return time, val


def get_atoms(directory, cell_list, nuc_list):
    """ Get total atom count as a function of time.

    Parameters
    ----------
    directory : str
        Directory to read results from.
    cell_list : List[int]
        List of cell IDs to extract data from.
    nuc_list : List[str]
        List of nuclides to extract data from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : Dict[Dict[np.array]]
        Total number of atoms, indexed [cell id : int][nuclide : str]
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = {}
    time = np.zeros(count)

    for cell in cell_list:
        val[cell] = {}
        for nuc in nuc_list:
            val[cell][nuc] = np.zeros(count)

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            handle = open(directory + '/' + file, 'rb')
            result = pickle.load(handle)
            handle.close()

            for cell in cell_list:
                if cell in result.num[0]:
                    for nuc in nuc_list:
                        if nuc in result.num[0][cell]:
                            val[cell][nuc][ind] = result.num[0][cell][nuc]
            time[ind] = result.time
    return time, val


def get_atoms_volaveraged(directory, op, cell_list, nuc_list):
    """ Get volume averaged atom count as a function of time.

    This function sums the atom concentration from each cell and then divides
    by the volume sum.

    Parameters
    ----------
    directory : str
        Directory to read results from.
    op : function.Operator
        The operator used in this simulation. Contains volumes.
    cell_list : List[int]
        List of cell IDs to average.
    nuc_list : List[str]
        List of nuclides to extract data from.

    Returns
    -------
    time : np.array
        Time for each step.
    val : Dict[np.array]
        Volume averaged atoms, indexed [nuclide : str]
    """

    # First, calculate how many step files are in the folder

    count = 0
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            count += 1

    # Allocate result
    val = {}
    time = np.zeros(count)

    for nuc in nuc_list:
        val[nuc] = np.zeros(count)

    # Calculate volume of cell_list
    vol = 0.0
    for cell in cell_list:
        if cell in op.geometry.volume:
            vol += op.geometry.volume[cell]

    # Read in file, get eigenvalue, close file
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'step*'):
            # Get ind (files will be found out of order)
            name = file.split(".")
            ind = int(name[0][4::])

            # Read file
            handle = open(directory + '/' + file, 'rb')
            result = pickle.load(handle)
            handle.close()

            for cell in cell_list:
                if cell in result.num[0]:
                    for nuc in nuc_list:
                        if nuc in result.num[0][cell]:
                            val[nuc][ind] += result.num[0][cell][nuc]/vol
            time[ind] = result.time
    return time, val
