""" The OpenMC wrapper module.

This module implements the OpenDeplete -> OpenMC linkage.
"""

from collections import OrderedDict
import concurrent.futures
import os
import random
from subprocess import call
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import openmc
from openmc.stats import Box

from .depletion_chain import matrix_wrapper
from .reaction_rates import ReactionRates


class Settings(object):
    """ The Settings class.

    This contains the parameters passed to the integrator.  This includes
    time stepping, power, etc.  It also contains how to run OpenMC, and what
    settings OpenMC needs to run.

    Attributes
    ----------
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
    dt_vec : numpy.array
        Array of time steps to take.
    tol : float
        Tolerance for adaptive time stepping.
    output_dir : str
        Path to output directory to save results.
    """

    def __init__(self):
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
        self.dt_vec = None
        self.tol = None
        self.output_dir = None


class Materials(object):
    """The Materials class.

    This contains dictionaries indicating which cells are to be filled with
    what number of atoms and what libraries.

    Attributes
    ----------
    inital_density : OrderedDict[OrderedDict[float]]
        Initial density of the simulation.  Indexed as
        initial_density[name of region : str][name of nuclide : str].
    temperature : OrderedDict[str]
        Temperature in Kelvin for each region.  Indexed as temperature[name
        of region : float].
    cross_sections : str
        Path to cross_sections.xml file.
    sab : OrderedDict[str]
        ENDF S(a,b) name for a region that needs S(a,b) data.  Indexed as
        sab[name of region : str].  Not set if no S(a,b) needed for region.
    burn : OrderedDict[bool]
        burn[name of region : str] = True if region needs to be in burn.

    """

    def __init__(self):
        self.initial_density = None
        self.temperature = None
        self.cross_sections = None
        self.sab = None
        self.burn = None


class Geometry(object):
    """The Geometry class.

    Contains all geometry- and materials-related components necessary for
    depletion.

    Parameters
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object.
    volume : OrderedDict[float]
        Given a material ID, gives the volume of said material.
    materials : openmc_wrapper.Materials
        Materials to be used for this simulation.

    Attributes
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object.
    volume : OrderedDict[float]
        Given a material ID, gives the volume of said material.
    materials : Materials
        Materials to be used for this simulation.
    seed : int
        The RNG seed used in last OpenMC run.
    number_density : OrderedDict of int to OrderedDict of str to float
        The number density of a nuclide in a material.  Indexed as
        number_density[material ID : int][nuclide : str].
    total_number : OrderedDict of int to OrderedDict of str to float
        The number density of a nuclide in a material multiplied by the volume
        of the material.  Indexed as total_number[material ID : int][nuclide :
        str].
    participating_nuclides : set of str
        A set listing all unique nuclides available from cross_sections.xml.
    nuc_list : list of str
        A list of all nuclide names. Used for sorting the simulation.
    burn_list : list of int
        A list of all material IDs to be burned.  Used for sorting the simulation.
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

    def __init__(self, geometry, volume, materials):
        self.geometry = geometry
        self.volume = volume
        self.materials = materials
        self.seed = 0
        self.number_density = OrderedDict()
        self.total_number = OrderedDict()
        self.participating_nuclides = None
        self.nuc_list = []
        self.burn_list = []
        self.chain = None
        self.reaction_rates = None
        self.power = None
        self.mat_name = OrderedDict()
        self.burn_mat_to_ind = OrderedDict()
        self.burn_nuc_to_ind = None

    def initialize(self):
        """ Initializes the geometry.

        After geometry, volume, and materials are set, this function completes
        the geometry.

        Parameters
        ----------
        settings : Settings
            Settings to initialize with.
        """

        # Clear out OpenMC
        clean_up_openmc()

        # Extract number densities from geometry
        self.extract_all_materials()

        # Load participating nuclides
        self.load_participating()

        # Create reaction rate tables
        self.initialize_reaction_rates()

        # Finally, calculate total number densities
        self.calculate_total_number()

    def extract_all_materials(self):
        """ Iterate through all cells, create number density vectors from mats."""
        mat_ind = 0

        # First, for each material, extract number density
        cells = self.geometry.get_all_material_cells()
        for cell in cells:
            name = cell.name
            number_densities, mat_ids = extract_openmc_materials(cell)

            for i, mat_id in enumerate(mat_ids):
                self.number_density[mat_id] = number_densities[i]
                self.mat_name[mat_id] = name

                if self.materials.burn[name]:
                    self.burn_list.append(mat_id)
                    self.burn_mat_to_ind[str(mat_id)] = mat_ind
                    mat_ind += 1

    def initialize_reaction_rates(self):
        """ Create reaction rates object. """
        self.reaction_rates = ReactionRates(
            self.burn_mat_to_ind,
            self.burn_nuc_to_ind,
            self.chain.react_to_ind)

    def function_evaluation(self, vec, settings, print_out=True):
        """ Runs a simulation.

        Parameters
        ----------
        vec : list of numpy.array
            Total atoms to be used in function.
        settings : Settings
            Settings to run the sim with.
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
        self.generate_materials_xml(settings.round_number)
        self.generate_tally_xml()
        self.generate_settings_xml(settings)

        time_start = time.time()

        # Run model
        call(settings.openmc_call)
        time_openmc = time.time()

        statepoint_name = "statepoint." + str(settings.batches) + ".h5"

        # Extract results
        k = self.unpack_tallies_and_normalize(statepoint_name, settings.power)

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

    def start(self):
        """ Creates initial files, and returns initial vector.

        Returns
        -------
        list of numpy.array
            Total density for initial conditions.
        """
        # Write geometry.xml
        self.geometry.export_to_xml()

        # Return number density vector
        return self.total_density_list()

    def generate_materials_xml(self, round_number):
        """ Creates materials.xml from self.number_density.

        Iterates through each material in self.number_density and creates the
        openmc material object to generate materials.xml.

        Parameters
        ----------
        round_number : bool
            Whether or not to round output to OpenMC to 8 digits.
            Useful in testing, as OpenMC is incredibly sensitive to exact values.
        """
        openmc.reset_auto_material_id()

        materials = []

        for key_mat in self.number_density:
            mat = openmc.Material(material_id=key_mat)

            mat_name = self.mat_name[key_mat]
            mat.temperature = self.materials.temperature[mat_name]

            for key_nuc in self.number_density[key_mat]:
                # Check if in participating nuclides
                if key_nuc in self.participating_nuclides:
                    val = 1.0e-24*self.number_density[key_mat][key_nuc]

                    if round_number:
                        val_magnitude = np.floor(np.log10(val))
                        val_scaled = val / 10**val_magnitude
                        val_round = round(val_scaled, 8)

                        val = val_round * 10**val_magnitude

                    mat.add_nuclide(key_nuc, val)
            mat.set_density(units='sum')

            if mat_name in self.materials.sab:
                mat.add_s_alpha_beta(self.materials.sab[mat_name])

            materials.append(mat)

        materials_file = openmc.Materials(materials)
        materials_file.cross_sections = self.materials.cross_sections
        materials_file.export_to_xml()

    def generate_settings_xml(self, settings):
        """ Generates settings.xml.

        This function creates settings.xml using the value of the settings
        variable.

        Parameters
        ----------
        settings : Settings
            Operator settings configuration.

        Todo
        ----
            Rewrite to generalize source box.
        """

        batches = settings.batches
        inactive = settings.inactive
        particles = settings.particles

        # Just a generic settings file to get it running.
        settings_file = openmc.Settings()
        settings_file.batches = batches
        settings_file.inactive = inactive
        settings_file.particles = particles
        settings_file.source = openmc.Source(space=Box(settings.lower_left,
                                                       settings.upper_right))
        settings_file.entropy_lower_left = settings.lower_left
        settings_file.entropy_upper_right = settings.upper_right
        settings_file.entropy_dimension = settings.entropy_dimension

        # Set seed
        if settings.constant_seed is not None:
            seed = settings.constant_seed
        else:
            seed = random.randint(1, sys.maxsize-1)

        self.seed = seed
        settings_file.seed = seed

        settings_file.export_to_xml()

    def generate_tally_xml(self):
        """ Generates tally.xml.

        Using information from self.depletion_chain as well as the nuclides
        currently in the problem, this function automatically generates a
        tally.xml for the simulation.
        """
        chain = self.chain

        # ----------------------------------------------------------------------
        # Create tallies for depleting regions
        tally_ind = 1
        mat_filter_dep = openmc.MaterialFilter(self.burn_list)
        tallies_file = openmc.Tallies()

        nuc_superset = set()

        for mat in self.burn_list:
            for key in self.number_density[mat]:
                # Check if in participating nuclides
                if key in self.participating_nuclides:
                    nuc_superset.add(key)

        # For each reaction in the chain, for each nuclide, and for each
        # cell, make a tally
        tally_dep = openmc.Tally(tally_id=tally_ind)
        for key in nuc_superset:
            if key in chain.nuclide_dict:
                tally_dep.add_nuclide(key)

        for reaction in chain.react_to_ind:
            tally_dep.add_score(reaction)

        tallies_file.add_tally(tally_dep)

        tally_dep.add_filter(mat_filter_dep)
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

        # An issue with concurrent.futures is that it is far easier to write a
        # map, so I need to concatenate the data into a single variable with
        # which a map works.
        input_list = []
        for mat in self.burn_mat_to_ind:
            mat_ind = self.burn_mat_to_ind[mat]
            input_list.append((self.chain, self.reaction_rates, mat_ind))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            matrices = executor.map(matrix_wrapper, input_list)

        return list(matrices)

    def calculate_total_number(self):
        """ Calculates the total number of atoms.

        Simply multiplies self.number_density[mat][nuclide] by
        self.volume[mat] and saves the value in
        self.total_number[mat][nuclide]
        """

        for mat in self.number_density:
            self.total_number[mat] = OrderedDict()
            for nuclide in self.number_density[mat]:
                value = self.number_density[mat][nuclide] * self.volume[mat]
                self.total_number[mat][nuclide] = value

    def total_density_list(self):
        """ Returns a list of total density lists.

        This list is in the exact same order as depletion_matrix_list, so that
        matrix exponentiation can be done easily.

        Returns
        -------
        list of numpy.array
            A list of np.arrays containing total atoms of each cell.
        """

        total_density = []

        for mat_i, mat in enumerate(self.burn_list):

            total_density.append([])

            # Get all nuclides that exist in both chain and total_number
            # in the order of chain
            for nuc in self.nuc_list:
                if nuc in self.total_number[mat]:
                    total_density[mat_i].append(self.total_number[mat][nuc])
                else:
                    total_density[mat_i].append(0.0)
            # Convert to np.array
            total_density[mat_i] = np.array(total_density[mat_i])

        return total_density

    def get_non_participating_nuc(self):
        """ Returns a nested dictionary of nuclides not participating.

        Returns
        -------
        not_participating : dict of str to dict of str to float
            Not participating nuclides, indexed by cell id and nuclide id.

        """

        not_participating = {}

        for mat in self.burn_list:

            not_participating[mat] = {}

            # Get all nuclides that don't exist in chain but do in total_number
            for nuc in self.total_number[mat]:
                if nuc not in self.nuc_list:
                    not_participating[mat][nuc] = self.total_number[mat][nuc]

        return not_participating

    def set_density(self, total_density):
        """ Sets density.

        Sets the density in the exact same order as total_density_list outputs,
        allowing for internal consistency

        Parameters
        ----------
        total_density : list of numpy.array
            Total atoms.
        """

        # First, ensure self.total_number is clear
        for mat in self.burn_list:
            for nuc in self.nuc_list:
                if nuc in self.total_number[mat]:
                    self.total_number[mat].pop(nuc, None)


        for mat_i, mat in enumerate(self.burn_list):

            # Update total_number first
            for i in range(self.n_nuc):
                # Don't add if zero, for performance reasons.
                if total_density[mat_i][i] != 0.0:
                    nuc = self.nuc_list[i]
                    # Add a "infinitely dilute" quantity if negative
                    if total_density[mat_i][i] >= 0.0:
                        self.total_number[mat][nuc] = total_density[mat_i][i]
                    else:
                        self.total_number[mat][nuc] = 1.0
                        print("WARNING! Negative Number Densities!")
                        print("Material id = ", mat_i, " nuclide ", nuc,
                              " number = ", total_density[mat_i][i])

            # Then update number_density
            for nuc in self.total_number[mat]:
                self.number_density[mat][nuc] = self.total_number[mat][nuc] \
                                                 / self.volume[mat]

    def fill_nuclide_list(self):
        """ Creates a list of nuclides in the order they will appear in vecs.

        All this is is the name of each nuclide in self.chain.nuclides.
        """

        self.nuc_list = []

        for nuc in self.chain.nuclides:
            self.nuc_list.append(nuc.name)

    def unpack_tallies_and_normalize(self, filename, new_power):
        """ Unpack tallies from OpenMC

        This function reads the tallies generated by OpenMC (from the tally.xml
        file generated in generate_tally_xml) normalizes them so that the total
        power generated is new_power, and then stores them in the reaction rate
        database.

        Parameters
        ----------
        filename : str
            The statepoint file to read from.
        new_power : float
            The target power in MeV/cm.

        Returns
        -------
        k : float
            Eigenvalue of the last simulation.

        Todo
        ----
            Provide units for new_power
        """
        statepoint = openmc.StatePoint(filename)

        k_combined = statepoint.k_combined[0]

        # Generate new power dictionary

        self.power = OrderedDict()

        # ---------------------------------------------------------------------
        # Unpack depletion list
        tally_dep = statepoint.get_tally(id=1)

        # Zero out reaction_rates
        self.reaction_rates[:, :, :] = 0.0

        df_tally = tally_dep.get_pandas_dataframe()
        # For each mat to be burned
        for mat_str in self.burn_mat_to_ind:
            mat = int(mat_str)
            df_mat = df_tally[df_tally["material"] == mat]

            # For each nuclide that was tallied
            for nuc in self.burn_nuc_to_ind:

                # If density = 0, there was no tally
                if nuc not in self.total_number[mat]:
                    continue

                nuclide = self.chain.nuc_by_ind(nuc)

                df_nuclide = df_mat[df_mat["nuclide"] == nuc]

                # For each reaction pathway
                for j in range(nuclide.n_reaction_paths):
                    # Extract tally
                    tally_type = nuclide.reaction_type[j]

                    k = self.reaction_rates.react_to_ind[tally_type]
                    value = df_nuclide[df_nuclide["score"] ==
                                       tally_type]["mean"].values[0]

                    # The reaction rates are normalized to total number of
                    # atoms in the simulation.
                    self.reaction_rates[mat_str, nuclide.name, k] = value \
                        / self.total_number[mat][nuc]

                    # Calculate power if fission
                    if tally_type == "fission":
                        power = value * nuclide.fission_power
                        if mat not in self.power:
                            self.power[mat] = power
                        else:
                            self.power[mat] += power

        # ---------------------------------------------------------------------
        # Normalize to power
        original_power = sum(self.power.values())

        self.reaction_rates[:, :, :] *= (new_power / original_power)

        return k_combined

    def load_participating(self):
        """ Loads a cross_sections.xml file to find participating nuclides.

        This allows for nuclides that are important in the decay chain but not
        important neutronically, or have no cross section data.
        """

        # Reads cross_sections.xml to create a dictionary containing
        # participating (burning and not just decaying) nuclides.

        filename = self.materials.cross_sections

        self.participating_nuclides = set()

        try:
            tree = ET.parse(filename)
        except:
            if filename is None:
                print("No cross_sections.xml specified in materials.")
            else:
                print("Cross section file \"", filename, "\" is invalid.")
            exit()

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

def extract_openmc_materials(cell):
    """ Extracts a dictionary from an OpenMC material object

    Parameters
    ----------
    cell : openmc.Cell
        The cell to extract from/

    Returns
    -------
    list of OrderedDict of str to float
        A list of ordered dictionaries containing the nuclides of interest.
    list of int
        IDs of the materials used.
    """

    result = []
    mat_id = []

    if isinstance(cell.fill, openmc.Material):
        nuc = OrderedDict()
        for nuclide in cell.fill.nuclides:
            name = nuclide[0].name
            number = nuclide[1] * 1.0e24
            nuc[name] = number
        result.append(nuc)
        mat_id.append(cell.fill.id)
    else:
        for mat in cell.fill:
            nuc = OrderedDict()
            for nuclide in mat.nuclides:
                name = nuclide[0].name
                number = nuclide[1] * 1.0e24
                nuc[name] = number
            result.append(nuc)
            mat_id.append(mat.id)

    return result, mat_id


def clean_up_openmc():
    """ Resets all automatic indexing in OpenMC, as these get in the way. """
    openmc.reset_auto_material_id()
    openmc.reset_auto_surface_id()
    openmc.reset_auto_cell_id()
    openmc.reset_auto_universe_id()
