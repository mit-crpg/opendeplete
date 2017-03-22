""" The OpenMC wrapper module.

This module implements the OpenDeplete -> OpenMC linkage.
"""

import copy
from collections import OrderedDict
import concurrent.futures
import os
from itertools import repeat
import random
from subprocess import call
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc
from openmc.stats import Box

from .atom_number import AtomNumber
from .depletion_chain import DepletionChain, matrix_wrapper
from .reaction_rates import ReactionRates
from .function import Settings, Operator

class OpenMCSettings(Settings):
    """ The OpenMCSettings class.

    Extends Settings to provide information OpenMC needs to run.

    Attributes
    ----------
    dt_vec : numpy.array
        Array of time steps to take. (From Settings)
    tol : float
        Tolerance for adaptive time stepping. (From Settings)
    output_dir : str
        Path to output directory to save results. (From Settings)
    chain_file : str
        Path to the depletion chain xml file.  Defaults to the environment
        variable "OPENDEPLETE_CHAIN" if it exists.
    openmc_call : list of str
        The command to be used with subprocess.call to run a simulation. If no
        arguments are to be passed, a string suffices.  To run with mpiexec,
        a list of strings is needed.
    particles : int
        Number of particles to simulate per batch.
    batches : int
        Number of batches.
    inactive : int
        Number of inactive batches.
    lower_left : list of float
        Coordinate of lower left of bounding box of geometry.
    upper_right : list of float
        Coordinate of upper right of bounding box of geometry.
    entropy_dimension : list of int
        Grid size of entropy.
    round_number : bool
        Whether or not to round output to OpenMC to 8 digits.
        Useful in testing, as OpenMC is incredibly sensitive to exact values.
    constant_seed : int
        If present, all runs will be performed with this seed.
    power : float
        Power of the reactor (currently in MeV/second-cm).
    """

    def __init__(self):
        Settings.__init__(self)
        # OpenMC specific
        try:
            self.chain_file = os.environ["OPENDEPLETE_CHAIN"]
        except KeyError:
            self.chain_file = None
        self.openmc_call = None
        self.particles = None
        self.batches = None
        self.inactive = None
        self.lower_left = None
        self.upper_right = None
        self.entropy_dimension = None

        # OpenMC testing specific
        self.round_number = False
        self.constant_seed = None

        # Depletion problem specific
        self.power = None

class Materials(object):
    """The Materials class.

    Contains information about cross sections for a cell.

    Attributes
    ----------
    temperature : float
        Temperature in Kelvin for each region.
    sab : str or list of str
        ENDF S(a,b) name for a region that needs S(a,b) data.  Not set if no
        S(a,b) needed for region.
    """

    def __init__(self):
        self.temperature = None
        self.sab = None


class OpenMCOperator(Operator):
    """The OpenMC Operator class.

    Provides Operator functions for OpenMC.

    Parameters
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object.
    settings : OpenMCSettings
        Settings object.

    Attributes
    ----------
    settings : OpenMCSettings
        Settings object. (From Operator)
    geometry : openmc.Geometry
        The OpenMC geometry object.
    materials : list of Materials
        Materials to be used for this simulation.
    seed : int
        The RNG seed used in last OpenMC run.
    number : AtomNumber
        Total number of atoms in simulation.
    participating_nuclides : set of str
        A set listing all unique nuclides available from cross_sections.xml.
    chain : DepletionChain
        The depletion chain information necessary to form matrices and tallies.
    reaction_rates : ReactionRates
        Reaction rates from the last operator step.
    power : OrderedDict of str to float
        Material-by-Material power.  Indexed by material ID.
    mat_name : OrderedDict of str to int
        The name of region each material is set to.  Indexed by material ID.
    burn_mat_to_id : OrderedDict of str to int
        Dictionary mapping material ID (as a string) to an index in reaction_rates.
    burn_nuc_to_id : OrderedDict of str to int
        Dictionary mapping nuclide name (as a string) to an index in
        reaction_rates.
    n_nuc : int
        Number of nuclides considered in the decay chain.

    """

    def __init__(self, geometry, settings):
        Operator.__init__(self, settings)
        self.geometry = geometry
        self.materials = []
        self.seed = 0
        self.number = None
        self.participating_nuclides = None
        self.reaction_rates = None
        self.power = None
        self.mat_name = OrderedDict()
        self.burn_mat_to_ind = OrderedDict()
        self.burn_nuc_to_ind = None

        # Read depletion chain
        self.chain = DepletionChain.xml_read(settings.chain_file)

        # Clear out OpenMC
        clean_up_openmc()

        # Extract number densities from geometry
        self.extract_all_materials()

        # Load participating nuclides
        self.load_participating()

        # Create reaction rate tables
        self.initialize_reaction_rates()

    def extract_all_materials(self):
        """ Iterate through all cells, create number density vectors from mats."""

        # Get a set of materials, nuclides
        mat_burn = set()
        mat_not_burn = set()
        nuc_set = set()

        volume = OrderedDict()

        # Iterate once through the geometry to allocate arrays
        cells = self.geometry.get_all_material_cells()
        for cell_id in cells:
            cell = cells[cell_id]
            name = cell.name

            if isinstance(cell.fill, openmc.Material):
                mat = cell.fill
                for nuclide in mat.get_nuclide_densities():
                    nuc_set.add(nuclide)
                if mat.depletable:
                    mat_burn.add(str(mat.id))
                    volume[str(mat.id)] = mat.volume
                else:
                    mat_not_burn.add(str(mat.id))
                self.mat_name[mat.id] = name
            else:
                for mat in cell.fill:
                    for nuclide in mat.get_nuclide_densities():
                        nuc_set.add(nuclide)
                    if mat.depletable:
                        mat_burn.add(str(mat.id))
                        volume[str(mat.id)] = mat.volume
                    else:
                        mat_not_burn.add(str(mat.id))
                    self.mat_name[mat.id] = name

        need_vol = []

        for mat_id in volume:
            if volume[mat_id] is None:
                need_vol.append(mat_id)

        if len(need_vol) > 0:
            exit("Need volumes for materials: " + str(need_vol))

        # Alphabetize the sets
        mat_burn = sorted(list(mat_burn))
        mat_not_burn = sorted(list(mat_not_burn))
        nuc_set = sorted(list(nuc_set))

        # Construct a global nuclide dictionary, burned first
        nuc_dict = copy.copy(self.chain.nuclide_dict)

        i = len(nuc_dict)

        for nuc in nuc_set:
            if nuc not in nuc_dict:
                nuc_dict[nuc] = i
                i += 1

        # Same with materials
        mat_dict = OrderedDict()
        self.burn_mat_to_ind = OrderedDict()
        i = 0
        for mat in mat_burn:
            mat_dict[mat] = i
            self.burn_mat_to_ind[mat] = i
            i += 1

        for mat in mat_not_burn:
            mat_dict[mat] = i
            i += 1

        n_mat_burn = len(mat_burn)
        n_nuc_burn = len(self.chain.nuclide_dict)

        self.number = AtomNumber(mat_dict, nuc_dict, volume, n_mat_burn, n_nuc_burn)

        self.materials = [None] * self.number.n_mat

        # Now extract the number densities and store
        cells = self.geometry.get_all_material_cells()
        for cell in cells.values():
            if isinstance(cell.fill, openmc.Material):
                self.set_number_from_mat(cell.fill)
            else:
                for mat in cell.fill:
                    self.set_number_from_mat(mat)

    def set_number_from_mat(self, mat):
        """ Extracts material and number densities from openmc.Material

        Parameters
        ----------
        mat : openmc.Materials
            The material to read from
        """

        mat_id = str(mat.id)
        mat_ind = self.number.mat_to_ind[mat_id]

        self.materials[mat_ind] = Materials()
        self.materials[mat_ind].sab = mat._sab
        self.materials[mat_ind].temperature = mat.temperature

        nuc_dens = mat.get_nuclide_atom_densities()
        for nuclide in nuc_dens:
            name = nuclide.name
            number = nuc_dens[nuclide][1] * 1.0e24
            self.number.set_atom_density(mat_id, name, number)

    def initialize_reaction_rates(self):
        """ Create reaction rates object. """
        self.reaction_rates = ReactionRates(
            self.burn_mat_to_ind,
            self.burn_nuc_to_ind,
            self.chain.react_to_ind)

        self.chain.nuc_to_react_ind = self.burn_nuc_to_ind

    def eval(self, vec, print_out=True):
        """ Runs a simulation.

        Parameters
        ----------
        vec : list of numpy.array
            Total atoms to be used in function.
        print_out : bool, optional
            Whether or not to print out time.

        Returns
        -------
        mat : list of scipy.sparse.csr_matrix
            Matrices for the next step.
        k : float
            Eigenvalue of the problem.
        rates : ReactionRates
            Reaction rates from this simulation.
        seed : int
            Seed for this simulation.
        """

        # Update status
        self.set_density(vec)

        # Recreate model
        self.generate_materials_xml()
        self.generate_tally_xml()
        self.generate_settings_xml()

        time_start = time.time()

        # Run model
        call(self.settings.openmc_call)
        time_openmc = time.time()

        statepoint_name = "statepoint.{}.h5".format(self.settings.batches)

        # Extract results
        k = self.unpack_tallies_and_normalize(statepoint_name)

        time_unpack = time.time()
        os.remove(statepoint_name)

        # Compute matrices
        mat = self.depletion_matrix_list()
        time_matrix = time.time()

        if print_out:
            print("Time to openmc: ", time_openmc - time_start)
            print("Time to unpack: ", time_unpack - time_openmc)
            print("Time to matrix: ", time_matrix - time_unpack)

        return mat, k, self.reaction_rates, self.seed

    def initial_condition(self):
        """ Performs final setup and returns initial condition.

        Returns
        -------
        list of numpy.array
            Total density for initial conditions.
        """

        # Write geometry.xml
        self.geometry.export_to_xml()

        # Return number density vector
        return self.total_density_list()

    def generate_materials_xml(self):
        """ Creates materials.xml from self.number_density.

        Iterates through each material in self.number_density and creates the
        openmc material object to generate materials.xml.
        """
        openmc.reset_auto_material_id()

        materials = []

        for key_mat in self.number.mat_to_ind:
            mat_id = self.number.mat_to_ind[key_mat]

            mat = openmc.Material(material_id=int(key_mat))

            mat.temperature = self.materials[mat_id].temperature
            mat._sab = self.materials[mat_id].sab

            for key_nuc in self.number.nuc_to_ind:
                # Check if in participating nuclides
                if key_nuc in self.participating_nuclides:
                    val = 1.0e-24*self.number.get_atom_density(key_mat, key_nuc)

                    # If nuclide is zero, do not add to the problem.
                    if val > 0.0:
                        if self.settings.round_number:
                            val_magnitude = np.floor(np.log10(val))
                            val_scaled = val / 10**val_magnitude
                            val_round = round(val_scaled, 8)

                            val = val_round * 10**val_magnitude

                        mat.add_nuclide(key_nuc, val)
                    else:
                        # Only output warnings if values are significantly
                        # negative.  CRAM does not guarantee positive values.
                        if val < -1.0e-21:
                            print("WARNING: nuclide ", key_nuc, " in material ", key_mat,
                                  " is negative (density = ", val, " at/barn-cm)")
                        self.number[key_mat, key_nuc] = 0.0

            mat.set_density(units='sum')

            materials.append(mat)

        materials_file = openmc.Materials(materials)
        materials_file.export_to_xml()

    def generate_settings_xml(self):
        """ Generates settings.xml.

        This function creates settings.xml using the value of the settings
        variable.

        Todo
        ----
            Rewrite to generalize source box.
        """

        batches = self.settings.batches
        inactive = self.settings.inactive
        particles = self.settings.particles

        # Just a generic settings file to get it running.
        settings_file = openmc.Settings()
        settings_file.batches = batches
        settings_file.inactive = inactive
        settings_file.particles = particles
        settings_file.source = openmc.Source(space=Box(self.settings.lower_left,
                                                       self.settings.upper_right))

        if self.settings.entropy_dimension is not None:
            entropy_mesh = openmc.Mesh()
            entropy_mesh.lower_left = self.settings.lower_left
            entropy_mesh.upper_right = self.settings.upper_right
            entropy_mesh.dimension = self.settings.entropy_dimension
            settings_file.entropy_mesh = entropy_mesh

        # Set seed
        if self.settings.constant_seed is not None:
            seed = self.settings.constant_seed
        else:
            seed = random.randint(1, sys.maxsize-1)

        settings_file.seed = self.seed = seed

        settings_file.export_to_xml()

    def generate_tally_xml(self):
        """ Generates tally.xml.

        Using information from self.depletion_chain as well as the nuclides
        currently in the problem, this function automatically generates a
        tally.xml for the simulation.
        """
        # ----------------------------------------------------------------------
        # Create tallies for depleting regions
        tally_ind = 1
        mat_filter_dep = openmc.MaterialFilter([int(id) for id in self.number.burn_mat_list])
        tallies_file = openmc.Tallies()

        nuc_superset = set()

        # Create the set of all nuclides in the decay chain in cells marked for
        # burning in which the number density is greater than zero.

        for nuc in self.number.nuc_to_ind:
            if nuc in self.participating_nuclides:
                if np.sum(self.number[:, nuc]) > 0.0:
                    nuc_superset.add(nuc)

        # For each reaction in the chain, for each nuclide, and for each
        # cell, make a tally
        tally_dep = openmc.Tally(tally_id=tally_ind)
        for key in nuc_superset:
            if key in self.chain.nuclide_dict:
                tally_dep.nuclides.append(key)

        for reaction in self.chain.react_to_ind:
            tally_dep.scores.append(reaction)

        tallies_file.append(tally_dep)

        tally_dep.filters.append(mat_filter_dep)
        tallies_file.export_to_xml()

    def depletion_matrix_list(self):
        """ Generates a list containing the depletion operators.

        Returns a list of sparse (CSR) matrices containing the depletion
        operators for this problem.  It is done in parallel using the
        concurrent futures package.

        Returns
        -------
        list of scipy.sparse.csr_matrix
            A list of sparse depletion matrices.

        Todo
        ----
            Generalize method away from process parallelism.
        """

        n_mat = len(self.burn_mat_to_ind)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            matrices = executor.map(matrix_wrapper, zip(
                repeat(self.chain, n_mat), self.reaction_rates.rates))

        return list(matrices)

    def total_density_list(self):
        """ Returns a list of total density lists.

        This list is in the exact same order as depletion_matrix_list, so that
        matrix exponentiation can be done easily.

        Returns
        -------
        list of numpy.array
            A list of np.arrays containing total atoms of each cell.
        """

        total_density = [self.number.get_mat_slice(i) for i in range(self.number.n_mat_burn)]

        return total_density

    def set_density(self, total_density):
        """ Sets density.

        Sets the density in the exact same order as total_density_list outputs,
        allowing for internal consistency

        Parameters
        ----------
        total_density : list of numpy.array
            Total atoms.
        """

        # Fill in values
        for i in range(self.number.n_mat_burn):
            self.number.set_mat_slice(i, total_density[i])

    def unpack_tallies_and_normalize(self, filename):
        """ Unpack tallies from OpenMC

        This function reads the tallies generated by OpenMC (from the tally.xml
        file generated in generate_tally_xml) normalizes them so that the total
        power generated is new_power, and then stores them in the reaction rate
        database.

        Parameters
        ----------
        filename : str
            The statepoint file to read from.

        Returns
        -------
        k : float
            Eigenvalue of the last simulation.

        Todo
        ----
            Provide units for power
        """

        self.reaction_rates[:, :, :] = 0.0

        file = h5py.File(filename, "r")

        k_combined = file["k_combined"][0]

        nuclides_binary = file["tallies/tally 1/nuclides"].value
        nuclides = [nuc.decode('utf8') for nuc in nuclides_binary]

        reactions_binary = file["tallies/tally 1/score_bins"].value
        reactions = [react.decode('utf8') for react in reactions_binary]

        # Get fast map
        nuc_ind = [self.reaction_rates.nuc_to_ind[nuc] for nuc in nuclides]
        react_ind = [self.reaction_rates.react_to_ind[react] for react in reactions]

        # Compute fission power
        # TODO : improve this calculation

        power = 0.0

        power_vec = np.zeros(self.reaction_rates.n_nuc)

        fission_ind = self.reaction_rates.react_to_ind["fission"]

        for nuclide in self.chain.nuclides:
            if nuclide.name in self.reaction_rates.nuc_to_ind:
                ind = self.reaction_rates.nuc_to_ind[nuclide.name]

                if 'fission' in nuclide.reaction_type:
                    j = nuclide.reaction_type.index('fission')
                    power_vec[ind] = nuclide.reaction_Q[j]*1e-6

        # Extract results
        for i, mat in enumerate(self.number.burn_mat_list):
            # Get material results hyperslab
            results = file["tallies/tally 1/results"][i, :, 0]

            results_expanded = np.zeros((self.reaction_rates.n_nuc, self.reaction_rates.n_react))
            number = np.zeros(self.reaction_rates.n_nuc)

            # Expand into our memory layout
            j = 0
            for nuc, i_nuc_results in zip(nuclides, nuc_ind):
                number[i_nuc_results] = self.number[mat, nuc]
                for react in react_ind:
                    results_expanded[i_nuc_results, react] = results[j]
                    j += 1

            # Add power
            power += np.dot(results_expanded[:, fission_ind], power_vec)

            # Divide by total number and store
            for i_nuc_results in nuc_ind:
                for react in react_ind:
                    if number[i_nuc_results] != 0.0:
                        results_expanded[i_nuc_results, react] /= number[i_nuc_results]

            self.reaction_rates.rates[i, :, :] = results_expanded

        self.reaction_rates[:, :, :] *= (self.settings.power / power)

        return k_combined

    def load_participating(self):
        """ Loads a cross_sections.xml file to find participating nuclides.

        This allows for nuclides that are important in the decay chain but not
        important neutronically, or have no cross section data.
        """

        # Reads cross_sections.xml to create a dictionary containing
        # participating (burning and not just decaying) nuclides.

        try:
            filename = os.environ["OPENMC_CROSS_SECTIONS"]
        except KeyError:
            filename = None

        self.participating_nuclides = set()

        try:
            tree = ET.parse(filename)
        except:
            if filename is None:
                print("No cross_sections.xml specified in materials.")
            else:
                print('Cross section file "', filename, '" is invalid.')
            raise

        root = tree.getroot()
        self.burn_nuc_to_ind = OrderedDict()
        nuc_ind = 0

        for nuclide_node in root.findall('library'):
            mats = nuclide_node.get('materials')
            if not mats:
                continue
            for name in mats.split():
                # Make a burn list of the union of nuclides in cross_sections.xml
                # and nuclides in depletion chain.
                if name not in self.participating_nuclides:
                    self.participating_nuclides.add(name)
                    if name in self.chain.nuclide_dict:
                        self.burn_nuc_to_ind[name] = nuc_ind
                        nuc_ind += 1

    @property
    def n_nuc(self):
        """Number of nuclides considered in the decay chain."""
        return len(self.chain.nuclides)

    def get_results_info(self):
        """ Returns volume list, cell lists, and nuc lists.

        Returns
        -------
        volume : list of float
            Volumes corresponding to materials in burn_list
        nuc_list : list of str
            A list of all nuclide names. Used for sorting the simulation.
        burn_list : list of int
            A list of all cell IDs to be burned.  Used for sorting the simulation.
        """

        nuc_list = self.number.burn_nuc_list
        burn_list = self.number.burn_mat_list

        volume = self.number.volume[0:self.number.n_mat_burn]

        return volume, nuc_list, burn_list

def density_to_mat(dens_dict):
    """ Generates an OpenMC material from a cell ID and self.number_density.
    Parameters
    ----------
    m_id : int
        Cell ID.
    Returns
    -------
    openmc.Material
        The OpenMC material filled with nuclides.
    """

    mat = openmc.Material()
    for key in dens_dict:
        mat.add_nuclide(key, 1.0e-24*dens_dict[key])
    mat.set_density('sum')

    return mat

def clean_up_openmc():
    """ Resets all automatic indexing in OpenMC, as these get in the way. """
    openmc.reset_auto_material_id()
    openmc.reset_auto_surface_id()
    openmc.reset_auto_cell_id()
    openmc.reset_auto_universe_id()
