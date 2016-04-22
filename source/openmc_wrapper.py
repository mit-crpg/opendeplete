""" The OpenMC wrapper module.

This module implements the OpenDeplete -> OpenMC linkage.
"""

import openmc
import os
import time
import reaction_rates
from subprocess import call
from results import *
from collections import OrderedDict
import depletion_chain
import numpy as np


class Settings:
    """ The Settings class.

    This contains the parameters passed to the integrator.  This includes
    time stepping, power, etc.  It also contains how to run OpenMC, and what
    settings OpenMC needs to run.

    Attributes
    ----------
    cross_sections : str
        Path to cross_sections.xml.
    chain_file : str
        Path to the depletion chain xml file.
    openmc_call : List[str]
        The command to be used with subprocess.call to run a simulation. If no
        arguments are to be passed, a string suffices.  To run with mpiexec,
        a list of strings is needed.
    particles : int
        Number of particles to simulate per batch.
    batches : int
        Number of batches.
    inactive : int
        Number of inactive batches.
    power : float
        Power of the reactor (currently in MeV/second-cm).
    dt_vec : numpy.array
        Array of time steps to take.
    output_dir : str
        Path to output directory to save results.
    """

    def __init__(self):
        # OpenMC specific
        self.cross_sections = None
        self.chain_file = None
        self.openmc_call = None
        self.particles = None
        self.batches = None
        self.inactive = None

        # Depletion problem specific
        self.power = None
        self.dt_vec = None
        self.output_dir = None


class Materials:
    """ The Materials class.

    This contains dictionaries indicating which cells are to be filled with
    what number of atoms and what libraries.

    Attributes
    ----------
    inital_density : OrderedDict[OrderedDict[float]]
        Initial density of the simulation.  Indexed as
        initial_density[name of region : str][name of nuclide : str].
    library : OrderedDict[str]
        ENDF tag for the library data (e.g. '71c').  Indexed as
        library[name of region : str].
    sab : OrderedDict[str]
        ENDF S(a,b) name for a region that needs S(a,b) data.  Indexed as
        sab[name of region : str].  Not set if no S(a,b) needed for region.
    library_sab : OrderedDict[str]
        ENDF tag for the S(a,b) data (e.g. '15t').  Indexed as
        library_sab[name of region : str].  Not set if no S(a,b) needed for
        region.
    burn : OrderedDict[bool]
        burn[name of region : str] = True if region needs to be in burn.
    """

    def __init__(self):
        self.initial_density = None
        self.library = None
        self.sab = None
        self.library_sab = None
        self.burn = None


class Geometry:
    """ The Geometry class.

    Contains all geometry- and materials-related components necessary for
    depletion.

    Attributes
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object.
    volume : OrderedDict[float]
        Given a cell ID, gives the volume of said cell.
    materials : openmc_wrapper.Materials
        Materials to be used for this simulation.
    seed : int
        The RNG seed used in last OpenMC run.
    number_density : OrderedDict[OrderedDict[float]]
        The number density of a nuclide in a cell.  Indexed as
        number_density[cell ID : int][nuclide : str].
    total_number : OrderedDict[OrderedDict[float]]
        The number density of a nuclide in a cell multiplied by the volume of
        the cell.  Indexed as total_number[cell ID : int][nuclide : str].
    participating_nuclides : Set[str]
        A set listing all unique nuclides available from cross_sections.xml.
    burn_list : List[int]
        A list of all cell IDs to be burned.  Used for sorting the simulation.
    chain : depletion_chain.DepletionChain
        The depletion chain information necessary to form matrices and tallies.
    reaction_rates : reaction_rates.ReactionRates
        Reaction rates from the last operator step.
    power : OrderedDict[float]
        Cell-by-Cell power.  Indexed by cell ID.
    mat_name : OrderedDict[str]
        The name of region each cell is set to.  Indexed by cell ID.
    """

    def __init__(self):
        self.geometry = None
        self.volume = None
        self.materials = None
        self.seed = None
        self.number_density = None
        self.total_number = None
        self.participating_nuclides = None
        self.burn_list = None
        self.chain = None
        self.reaction_rates = None
        self.power = None
        self.mat_name = None

    def initialize(self, settings):
        """ Initializes the geometry.

        After geometry, volume, and materials are set, this function completes
        the geometry.

        Parameters
        ----------
        settings : openmc_wrapper.Settings
            Settings to initialize with.
        """

        import copy

        # Clear out OpenMC
        openmc.reset_auto_material_id()
        openmc.reset_auto_surface_id()
        openmc.reset_auto_cell_id()
        openmc.reset_auto_universe_id()

        self.number_density = OrderedDict()
        self.mat_name = OrderedDict()
        self.burn_list = []

        # First, for each cell, create an entry in number_density
        cells = self.geometry.root_universe.get_all_cells()
        for cid in cells:
            cell = cells[cid]
            name = cell.name
            self.mat_name[cid] = name
            # Create entry in self.number_density using initial_density
            self.number_density[cid] = \
                copy.deepcopy(self.materials.initial_density[name])
            # Fill with a material (linked to cell id)
            cell.fill = self.density_dictionary_to_openmc_mat(cid)
            # If should burn, add to burn list:
            if self.materials.burn[name]:
                self.burn_list.append(cid)

        # Then, write geometry.xml
        geometry_file = openmc.GeometryFile()
        geometry_file.geometry = self.geometry
        geometry_file.export_to_xml()

        # Load participating nuclides
        self.load_participating(settings.cross_sections)

        # Create reaction rate tables
        self.reaction_rates = OrderedDict()
        for cell in self.burn_list:
            self.reaction_rates[cell] = \
                reaction_rates.ReactionRates(self.chain)

        # Finally, calculate total number densities
        self.total_number = OrderedDict()
        self.calculate_total_number()

    def function_evaluation(self, vec, settings):
        """ Runs a simulation.

        Parameters
        ----------
        vec : List[numpy.array]
            Total atoms to be used in function.
        settings : openmc_wrapper.Settings
            Settings to run the sim with.

        Returns
        -------
        mat : List[scipy.sparse.csr_matrix]
            Matrices for the next step.
        k : float
            Eigenvalue of the problem.
        rates : reaction_rates.ReactionRates
            Reaction rates from this simulation.
        seed : int
            Seed for this simulation.
        """

        # Update status
        self.set_density(vec)

        # Recreate model
        self.generate_materials_xml()
        self.generate_tally_xml()
        self.generate_settings_xml(settings)

        # Run model
        devnull = open(os.devnull, 'w')

        t1 = time.time()
        call(settings.openmc_call)

        statepoint_name = "statepoint." + str(settings.batches) + ".h5"

        # Extract results
        t2 = time.time()
        k = self.unpack_tallies_and_normalize(statepoint_name, settings.power)
        t3 = time.time()
        os.remove(statepoint_name)
        mat = self.depletion_matrix_list()
        t4 = time.time()

        rates = extract_rates(self)

        print("Time to openmc: ", t2-t1)
        print("Time to unpack: ", t3-t2)
        print("Time to matrix: ", t4-t3)

        return mat, k, rates, self.seed

    def start(self):
        """ Creates initial files, and returns initial vector.

        Returns
        -------
        list[numpy.array]
            Total density for initial conditions.
        """
        # Write geometry.xml
        geometry_file = openmc.GeometryFile()
        geometry_file.geometry = self.geometry
        geometry_file.export_to_xml()

        # Return number density vector
        return self.total_density_list()

    def generate_materials_xml(self):
        """ Creates materials.xml from self.number_density.

        Iterates through each cell in self.number_density and creates the
        openmc material object to generate materials.xml.
        """
        openmc.reset_auto_material_id()

        mat = []
        i = 0

        for key_mat in self.number_density:
            mat.append(openmc.Material(material_id=key_mat))

            mat_name = self.mat_name[key_mat]

            total = 0.0
            for key_nuc in self.number_density[key_mat]:
                # Check if in participating nuclides
                if key_nuc in self.participating_nuclides:
                    nuc = openmc.Nuclide(key_nuc,
                                         xs=self.materials.library[mat_name])
                    mat[i].add_nuclide(nuc,
                                       self.number_density[key_mat][key_nuc])
                    total += self.number_density[key_mat][key_nuc]
            mat[i].set_density('atom/cm3', total)

            if mat_name in self.materials.sab:
                mat[i].add_s_alpha_beta(self.materials.sab[mat_name],
                                        self.materials.library_sab[mat_name])

            i += 1

        materials_file = openmc.MaterialsFile()
        materials_file.add_materials(mat)
        materials_file.export_to_xml()

    def generate_settings_xml(self, settings):
        """ Generates settings.xml.

        This function creates settings.xml using the value of the settings
        variable.

        Parameters
        ----------
        settings : openmc_wrapper.Settings
            Operator settings configuration.

        Todo
        ----
            Rewrite to generalize source box.
        """
        import random
        import sys
        from openmc.source import Source
        from openmc.stats import Box
        pitch = 1.26197

        batches = settings.batches
        inactive = settings.inactive
        particles = settings.particles

        # Just a generic settings file to get it running.
        settings_file = openmc.SettingsFile()
        settings_file.batches = batches
        settings_file.inactive = inactive
        settings_file.particles = particles
        settings_file.source = Source(space=Box([-0.0, -0.0, -1],
                                                [3/2*pitch, 3/2*pitch, 1]))
        settings_file.entropy_lower_left = [-0.0, -0.0, -1.e50]
        settings_file.entropy_upper_right = [3/2*pitch, 3/2*pitch, 1.e50]
        settings_file.entropy_dimension = [10, 10, 1]

        # Set seed
        seed = random.randint(1, sys.maxsize-1)
        seed = 1
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
        cell_filter_dep = openmc.Filter(type='cell', bins=self.burn_list)
        tallies_file = openmc.TalliesFile()

        nuc_superset = set()

        for cell in self.burn_list:
            for key in self.number_density[cell]:
                # Check if in participating nuclides
                if key in self.participating_nuclides:
                    nuc_superset.add(key)

        # For each reaction in the chain, for each nuclide, and for each
        # cell, make a tally
        tally_dep = openmc.Tally(tally_id=tally_ind)
        for key in nuc_superset:
            if key in chain.nuclide_dict:
                tally_dep.add_nuclide(key)

        for reaction in chain.reaction_list:
            tally_dep.add_score(reaction)

        tallies_file.add_tally(tally_dep)

        tally_dep.add_filter(cell_filter_dep)
        tallies_file.export_to_xml()

    def depletion_matrix_list(self):
        """ Generates a list containing the depletion operators.

        Returns a list of sparse (CSR) matrices containing the depletion
        operators for this problem.  It is done in parallel using the
        concurrent futures package.

        Returns
        -------
        List[scipy.sparse.csr_matrix]
            A list of sparse depletion matrices.

        Todo
        ----
            Generalize method away from process parallelism.
        """
        import concurrent.futures

        # An issue with concurrent.futures is that it is far easier to write a
        # map, so I need to concatenate the data into a single variable with
        # which a map works.
        input_list = []
        for rate in self.reaction_rates:
            input_list.append((self.chain, self.reaction_rates[rate]))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            matrices = executor.map(depletion_chain.matrix_wrapper, input_list)

        return list(matrices)

    def density_dictionary_to_openmc_mat(self, m_id):
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

        mat = openmc.Material(material_id=m_id)
        total = 0.0
        for key in self.number_density[m_id]:
            nuc = openmc.Nuclide(key)
            mat.add_nuclide(nuc, self.number_density[m_id][key])
            total += self.number_density[m_id][key]
        mat.set_density('atom/cm3', total)

        return mat

    def calculate_total_number(self):
        """ Calculates the total number of atoms.

        Simply multiplies self.number_density[cell][nuclide] by
        self.volume[cell] and saves the value in
        self.total_number[cell][nuclide]
        """

        for cell in self.number_density:
            self.total_number[cell] = OrderedDict()
            for nuclide in self.number_density[cell]:
                value = self.number_density[cell][nuclide] * self.volume[cell]
                self.total_number[cell][nuclide] = value

    def total_density_list(self):
        """ Returns a list of total density lists.

        This list is in the exact same order as depletion_matrix_list, so that
        matrix exponentiation can be done easily.

        Returns
        -------
        List[numpy.array]
            A list of np.arrays containing total atoms of each cell.

        Todo
        ----
            Make this method less fragile.  The only thing guaranteeing the
            order of vectors and matrices is self.burn_list's order.
        """

        total_density = []

        cell_i = 0

        for cell in self.burn_list:

            total_density.append([])

            # Get all nuclides that exist in both chain and total_number
            # in the order of chain
            for i in range(len(self.chain.nuclides)):
                if self.chain.nuclides[i].name in self.total_number[cell]:
                    total_density[cell_i].append(
                        self.total_number[cell][self.chain.nuclides[i].name])
                else:
                    total_density[cell_i].append(0.0)
            # Convert to np.array
            total_density[cell_i] = np.array(total_density[cell_i])
            cell_i += 1

        return total_density

    def set_density(self, total_density):
        """ Sets density.

        Sets the density in the exact same order as total_density_list outputs,
        allowing for internal consistency

        Parameters
        ----------
        total_density : list[numpy.array]
            Total atoms.

        Todo
        ----
            Make this method less fragile.  The only thing guaranteeing the
            order of vectors and matrices is self.burn_list's order.
        """

        cell_i = 0

        for cell in self.burn_list:

            # Update total_number first
            for i in range(len(self.chain.nuclides)):
                # Don't add if zero, for performance reasons.
                if total_density[cell_i][i] != 0.0:
                    nuc = self.chain.nuclides[i].name
                    # Add a "infinitely dilute" quantity if negative
                    # TODO: DEBUG
                    if total_density[cell_i][i] > 0.0:
                        self.total_number[cell][nuc] = total_density[cell_i][i]
                    else:
                        self.total_number[cell][nuc] = 1.0e5

            cell_i += 1

            # Then update number_density
            for nuc in self.total_number[cell]:
                self.number_density[cell][nuc] = self.total_number[cell][nuc] \
                                                 / self.volume[cell]

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
        import openmc.statepoint as sp

        statepoint = sp.StatePoint(filename)

        # Link with summary file so that cell IDs work.
        su = openmc.Summary('summary.h5')
        statepoint.link_with_summary(su)

        k = statepoint.k_combined[0]

        # Generate new power dictionary

        self.power = OrderedDict()

        # ---------------------------------------------------------------------
        # Unpack depletion list
        tally_dep = statepoint.get_tally(id=1)

        df = tally_dep.get_pandas_dataframe()
        for i in range(len(self.burn_list)):
            cell = self.burn_list[i]

            df_cell = df[df["cell"] == cell]

            # For each nuclide that is in the depletion chain
            for key in self.total_number[cell]:
                if key in self.chain.nuclide_dict:
                    # Check if in participating nuclides
                    if key in self.participating_nuclides:
                        # Pull out nuclide object to iterate over reaction rate
                        nuclide = self.chain.nuc_by_ind(key)

                        df_nuclide = df_cell[df_cell["nuclide"] == key]

                        for j in range(nuclide.n_reaction_paths):
                            tally_type = nuclide.reaction_type[j]
                            value = df_nuclide[df_nuclide["score"] ==
                                               tally_type]["mean"].values[0]

                            self.reaction_rates[cell].rate[nuclide.name][j] = \
                                value

                            # Calculate power if fission
                            if tally_type == "fission":
                                power = value * nuclide.fission_power
                                if cell not in self.power:
                                    self.power[cell] = power
                                else:
                                    self.power[cell] += power
                    else:
                        # Set reaction rates to zero
                        nuclide = self.chain.nuc_by_ind(key)

                        df_nuclide = df_cell[df_cell["nuclide"] == key]

                        for j in range(nuclide.n_reaction_paths):
                            self.reaction_rates[cell].rate[nuclide.name][j] = \
                                0.0

        # ---------------------------------------------------------------------
        # Normalize to power
        original_power = sum(self.power.values())

        test = 0.0

        for i in range(len(self.burn_list)):
            cell = self.burn_list[i]

            # For each nuclide that is in the depletion chain
            for key in self.total_number[cell]:
                if key in self.chain.nuclide_dict:
                    nuclide = self.chain.nuc_by_ind(key)

                    for j in range(nuclide.n_reaction_paths):

                        if self.number_density[cell][key] != 0:
                            # Normalize reaction rates and divide out number of
                            # nuclides, yielding cross section.
                            self.reaction_rates[cell].rate[nuclide.name][j] \
                                *= (new_power / original_power)
                            self.reaction_rates[cell].rate[nuclide.name][j] \
                                /= self.total_number[cell][key]

        return k

    def load_participating(self, filename):
        """ Loads a cross_sections.xml file to find participating nuclides.

        This allows for nuclides that are important in the decay chain but not
        important neutronically, or have no cross section data.

        Parameters
        ----------
        filename : str
            Path to cross_sections.xml
        """
        import xml.etree.ElementTree as ET

        # Reads cross_sections.xml to create a dictionary containing
        # participating (burning and not just decaying) nuclides.
        self.participating_nuclides = set()

        tree = ET.parse(filename)
        root = tree.getroot()

        for nuclide_node in root.findall('ace_table'):
            name = nuclide_node.get('alias')
            if not name:
                continue
            name_parts = name.split(".")

            self.participating_nuclides.add(name_parts[0])
