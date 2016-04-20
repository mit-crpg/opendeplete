import numpy as np
import os
import fnmatch
import pickle

def get_eigval(directory):
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

def get_atoms_volaveraged(directory, geo, cell_list, nuc_list):
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
        if cell in geo.volume:
            vol += geo.volume[cell]

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
