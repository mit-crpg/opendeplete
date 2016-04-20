"""Geometry module.

This module contains information about the geometry necessary for the depletion
process and the OpenMC input file generation process.  Currently, the geometry
is hard-coded.  It is anticipated that this will change.

Example:
    To create a geometry, just instantiate it::

        $ geo = Geometry()
"""

import depletion_chain
import reaction_rates
import openmc
from collections import OrderedDict


class Geometry:
    def __init__(self):
        self.geometry = None
        """openmc.Geometry: The OpenMC geometry object."""
        self.initial_density = None
        """OrderedDict: Initial simulation number density."""
        self.number_density = None
        """OrderedDict: Current number density."""
        self.total_number = None
        """OrderedDict: Number density times volume."""
        self.should_deplete = None
        """OrderedDict: Dictionary of which cells should be depleted."""
        self.burn_list = None
        """list: List of which cells should be burned."""
        self.fuel_list = None
        """list: List of which cells contain fuel."""
        self.volume = None
        """OrderedDict: Volumes indexed by cell IDs."""
        self.depletion_chain = None
        """DepletionChain: Depletion chain object."""
        self.reaction_rates = None
        """ReactionRates: Reaction rates from last operator step."""
        self.power = None
        """OrderedDict: Nuclide-by-nuclide power."""
        self.material = None
        """OrderedDict: Material name from initial densities."""
        self.library = None
        """OrderedDict: Dictionary indicating what library to use for a cell."""
        self.sab = None
        """OrderedDict: Dictionary indicating if a cell needs SAB data."""
        self.library_sab = None
        """OrderedDict: Dictionary indicating what SAB library to use for a cell."""
        self.participating_nuclides = None
        """set: A set of nuclide names in cross_sections.xml."""
        self.seed = None
        """int: RNG seed for simulation."""

    def generate_geometry_xml(self):
        """ Creates geometry and geometry.xml.

        This function creates the initial geometry, a 4 pin reflective problem.
        One pin, containing gadolinium, is discretized into 5 radial cells of
        equal volume.  Reflections go through the center of this pin.

        Todo:
            Rewrite to allow for arbitrary geometry.
            Rewrite to be a shim between a neutronics code and the depletion
            code.
        """
        import math
        import numpy as np

        pitch = 1.26197
        r_fuel = 0.412275
        r_gap = 0.418987
        r_clad = 0.476121
        n_rings = 5

        # Calculate all the volumes of interest ahead of time
        v_fuel = math.pi * r_fuel**2
        v_gap = math.pi * r_gap**2 - v_fuel
        v_clad = math.pi * r_clad**2 - v_fuel - v_gap
        v_ring = v_fuel / n_rings

        # Form dictionaries for later use
        # Ordered dictionaries are used, because I need to guarantee that
        # when these are converted into linalg components, they remap
        # correctly.
        self.number_density = OrderedDict()
        self.should_deplete = OrderedDict()
        self.is_water = OrderedDict()
        self.is_fuel = OrderedDict()
        self.volume = OrderedDict()
        self.total_number = OrderedDict()
        self.total_number_vec = OrderedDict()
        self.material = OrderedDict()

        # Calculate pin discretization radii
        r_rings = np.zeros(n_rings)

        # Remaining rings
        for i in range(n_rings):
            r_rings[i] = math.sqrt(1.0/(math.pi) * v_ring * (i+1))

        # Form bounding box
        left = openmc.XPlane(x0=0, name='left')
        right = openmc.XPlane(x0=3/2*pitch, name='right')
        bottom = openmc.YPlane(y0=0, name='bottom')
        top = openmc.YPlane(y0=3/2*pitch, name='top')

        left.boundary_type = 'reflective'
        right.boundary_type = 'reflective'
        top.boundary_type = 'reflective'
        bottom.boundary_type = 'reflective'

        # ----------------------------------------------------------------------
        # Fill pin 1 (the one with gadolinium)
        gd_fuel_r = [openmc.ZCylinder(x0=0, y0=0, R=r_rings[i]) for i in range(n_rings)]
        gd_clad_ir = openmc.ZCylinder(x0=0, y0=0, R=r_gap)
        gd_clad_or = openmc.ZCylinder(x0=0, y0=0, R=r_clad)

        # Fuel
        gd_fuel_cell = [openmc.Cell() for i in range(n_rings)]

        gd_fuel_cell[0].region = -gd_fuel_r[0] & +left & +bottom
        self.should_deplete[gd_fuel_cell[0].id] = True
        self.is_fuel[gd_fuel_cell[0].id] = True
        self.volume[gd_fuel_cell[0].id] = v_ring / 4.0
        self.material[gd_fuel_cell[0].id] = 'fuel_gd'
        self.set_initial_density(gd_fuel_cell[0].id)

        for i in range(n_rings-1):
            gd_fuel_cell[i+1].region = +gd_fuel_r[i] & -gd_fuel_r[i+1] & +left & +bottom
            self.should_deplete[gd_fuel_cell[i+1].id] = True
            self.is_fuel[gd_fuel_cell[i+1].id] = True
            self.material[gd_fuel_cell[i+1].id] = 'fuel_gd'
            self.set_initial_density(gd_fuel_cell[i+1].id)
            self.volume[gd_fuel_cell[i+1].id] = v_ring / 4.0

        for i in range(n_rings):
            gd_fuel_cell[i].fill = self.density_dictionary_to_openmc_mat(gd_fuel_cell[i].id)

        # Gap
        gd_fuel_gap = openmc.Cell()
        gd_fuel_gap.region = +gd_fuel_r[n_rings-1] & -gd_clad_ir & +left & +bottom
        self.material[gd_fuel_gap.id] = 'gap'
        self.set_initial_density(gd_fuel_gap.id)
        gd_fuel_gap.fill = self.density_dictionary_to_openmc_mat(gd_fuel_gap.id)
        self.should_deplete[gd_fuel_gap.id] = False
        self.is_fuel[gd_fuel_gap.id] = False
        self.volume[gd_fuel_gap.id] = v_gap / 4.0

        # Clad
        gd_fuel_clad = openmc.Cell()
        gd_fuel_clad.region = +gd_clad_ir & -gd_clad_or & +left & +bottom
        self.material[gd_fuel_clad.id] = 'clad'
        self.set_initial_density(gd_fuel_clad.id)
        gd_fuel_clad.fill = self.density_dictionary_to_openmc_mat(gd_fuel_clad.id)
        self.should_deplete[gd_fuel_clad.id] = False
        self.is_fuel[gd_fuel_clad.id] = False
        self.volume[gd_fuel_clad.id] = v_clad / 4.0

        # ----------------------------------------------------------------------
        # Fill pin 2, 3 and 4 (without gadolinium)
        coords = [[pitch, 0], [0, pitch], [pitch, pitch]]

        fuel_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_fuel) for x in coords]
        clad_ir_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_gap) for x in coords]
        clad_or_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_clad) for x in coords]

        fuel_cell = [openmc.Cell() for x in coords]
        clad_cell = [openmc.Cell() for x in coords]
        gap_cell = [openmc.Cell() for x in coords]

        fuel_cell[0].region = -fuel_s[0] & +bottom
        fuel_cell[1].region = -fuel_s[1] & +left
        fuel_cell[2].region = -fuel_s[2]

        gap_cell[0].region = +fuel_s[0] & -clad_ir_s[0] & +bottom
        gap_cell[1].region = +fuel_s[1] & -clad_ir_s[1] & +left
        gap_cell[2].region = +fuel_s[2] & -clad_ir_s[2]

        clad_cell[0].region = +clad_ir_s[0] & -clad_or_s[0] & +bottom
        clad_cell[1].region = +clad_ir_s[1] & -clad_or_s[1] & +left
        clad_cell[2].region = +clad_ir_s[2] & -clad_or_s[2]

        self.volume[fuel_cell[0].id] = v_fuel / 2.0
        self.volume[fuel_cell[1].id] = v_fuel / 2.0
        self.volume[fuel_cell[2].id] = v_fuel

        self.volume[gap_cell[0].id] = v_gap / 2.0
        self.volume[gap_cell[1].id] = v_gap / 2.0
        self.volume[gap_cell[2].id] = v_gap

        self.volume[clad_cell[0].id] = v_clad / 2.0
        self.volume[clad_cell[1].id] = v_clad / 2.0
        self.volume[clad_cell[2].id] = v_clad

        for i in range(3):
            self.material[fuel_cell[i].id] = 'fuel'
            self.set_initial_density(fuel_cell[i].id)
            self.should_deplete[fuel_cell[i].id] = True
            self.is_fuel[fuel_cell[i].id] = True
            fuel_cell[i].fill = self.density_dictionary_to_openmc_mat(fuel_cell[i].id)

            self.material[gap_cell[i].id] = 'gap'
            self.set_initial_density(gap_cell[i].id)
            self.should_deplete[gap_cell[i].id] = False
            self.is_fuel[gap_cell[i].id] = False
            gap_cell[i].fill = self.density_dictionary_to_openmc_mat(gap_cell[i].id)

            self.material[clad_cell[i].id] = 'clad'
            self.set_initial_density(clad_cell[i].id)
            self.should_deplete[clad_cell[i].id] = False
            self.is_fuel[clad_cell[i].id] = False
            clad_cell[i].fill = self.density_dictionary_to_openmc_mat(clad_cell[i].id)

        # ----------------------------------------------------------------------
        # Fill coolant

        cool_cell = openmc.Cell()
        cool_cell.region = +clad_or_s[0] & +clad_or_s[1] & +clad_or_s[2] & +gd_clad_or & +left & -right & +bottom & -top
        self.material[cool_cell.id] = 'cool'
        self.set_initial_density(cool_cell.id)
        self.should_deplete[cool_cell.id] = False
        self.is_fuel[cool_cell.id] = False
        self.volume[cool_cell.id] = (3/2 * pitch)**2 - 2.25 * v_fuel - 2.25 * v_gap - 2.25 * v_clad
        self.is_water[cool_cell.id] = True
        cool_cell.fill = self.density_dictionary_to_openmc_mat(cool_cell.id)

        # ----------------------------------------------------------------------
        # Finalize geometry
        root = openmc.Universe(universe_id=0, name='root universe')

        root.add_cells([cool_cell] + clad_cell + gap_cell + fuel_cell + gd_fuel_cell + [gd_fuel_clad] + [gd_fuel_gap])

        self.geometry = openmc.Geometry()
        self.geometry.root_universe = root

        geometry_file = openmc.GeometryFile()
        geometry_file.geometry = self.geometry
        geometry_file.export_to_xml()

        # ----------------------------------------------------------------------
        # Create burn list from dictionaries

        self.burn_list = [key for key in self.should_deplete if self.should_deplete[key] is True]

        self.fuel_list = [key for key in self.is_fuel if self.is_fuel[key] is True]

        self.calculate_total_number()

    def set_initial_density(self, cell_id):
        """ Initializes self.number_density.

        This function moves the self.initial_density variable to the
        self.number_density array for use in simulation. self.initial_density
        is not touched during a simulation, but self.number_density is.

        Args:
            cell_id (int): The cell to copy densities for.
        """
        import copy
        self.number_density[cell_id] = copy.copy(self.initial_density[self.material[cell_id]])

    def generate_materials_xml(self):
        """ Creates materials.xml from self.number_density.

        Iterates through each cell in self.number_density and creates the openmc
        material object to generate materials.xml.
        """
        openmc.reset_auto_material_id()

        mat = []
        i = 0

        for key_mat in self.number_density:
            mat.append(openmc.Material(material_id=key_mat))

            mat_name = self.material[key_mat]

            total = 0.0
            for key_nuc in self.number_density[key_mat]:
                # Check if in participating nuclides
                if key_nuc in self.participating_nuclides:
                    nuc = openmc.Nuclide(key_nuc, xs=self.library[mat_name])
                    mat[i].add_nuclide(nuc, self.number_density[key_mat][key_nuc])
                    total += self.number_density[key_mat][key_nuc]
            mat[i].set_density('atom/cm3', total)

            if mat_name in self.sab:
                mat[i].add_s_alpha_beta(self.sab[mat_name], self.library_sab[mat_name])

            i += 1

        materials_file = openmc.MaterialsFile()
        materials_file.add_materials(mat)
        materials_file.export_to_xml()

    def generate_settings_xml(self, settings):
        """ Generates settings.xml.

        This problem has a hard-coded 100 batches, with 40 inactive batches. The
        number of neutrons per batch, however, is not hardcoded.

        Todo:
            Rewrite to generalize settings.

        Args:
            npb (int): Number of neutrons per batch to use.
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
        settings_file.source = Source(space=Box([-0.0, -0.0, -1], [3/2*pitch, 3/2*pitch, 1]))
        settings_file.entropy_lower_left = [-0.0, -0.0, -1.e50]
        settings_file.entropy_upper_right = [3/2*pitch, 3/2*pitch, 1.e50]
        settings_file.entropy_dimension = [10, 10, 1]

        # Set seed
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
        chain = self.depletion_chain

        # ----------------------------------------------------------------------
        # Create tallies for depleting regions
        tally_ind = 1
        cell_filter_dep = openmc.Filter(type='cell', bins=self.burn_list)
        extra_list = list(set(self.fuel_list) - set(self.burn_list))
        cell_filter_heat = openmc.Filter(type='cell', bins=extra_list)
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

        # ----------------------------------------------------------------------
        # Create tallies for non-depleting regions (hot fuel)
        tally_ind = 2
        nuc_superset = set()

        for cell in extra_list:
            for key in self.number_density[cell]:
                # Check if in participating nuclides
                if key in self.participating_nuclides:
                    nuc_superset.add(key)

        # For this, we tally fission reaction rates in all fuel regions.
        tally_heat = openmc.Tally(tally_id=tally_ind)
        for key in nuc_superset:
            if key in chain.nuclide_dict:
                k = chain.nuclide_dict[key]
                nuclide = chain.nuclides[k]
                if 'fission' in nuclide.reaction_type:
                    tally_heat.add_nuclide(key)

        tally_heat.add_score('fission')

        tallies_file.add_tally(tally_heat)
        tally_dep.add_filter(cell_filter_dep)
        tally_heat.add_filter(cell_filter_heat)
        tallies_file.export_to_xml()

    def calculate_total_number(self):
        """ Calculates the total number of atoms.

        This problem has a hard-coded 100 batches, with 40 inactive batches. The
        number of neutrons per batch, however, is not hardcoded.

        Args:
            npb (int): Number of neutrons per batch to use.
        """

        for cell in self.number_density:
            self.total_number[cell] = OrderedDict()
            for nuclide in self.number_density[cell]:
                value = self.number_density[cell][nuclide] * self.volume[cell]
                self.total_number[cell][nuclide] = value

    def load_depletion_data(self, filename):
        """ Load self.depletion_data

        This problem has a hard-coded 100 batches, with 40 inactive batches. The
        number of neutrons per batch, however, is not hardcoded.

        Args:
            npb (int): Number of neutrons per batch to use.
        """
        # Create a depletion chain object, and then allocate the reaction rate objects
        self.depletion_chain = depletion_chain.DepletionChain()
        self.depletion_chain.xml_read(filename)

        self.reaction_rates = OrderedDict()
        for cell in self.burn_list:
            self.reaction_rates[cell] = reaction_rates.ReactionRates(self.depletion_chain)

    def unpack_tallies_and_normalize(self, filename, new_power):
        """ Unpack tallies from OpenMC

        This function reads the tallies generated by OpenMC (from the tally.xml
        file generated in generate_tally_xml) normalizes them so that the total
        power generated is new_power, and then stores them in the reaction rate
        database.

        Args:
            filename (str): The statepoint file to read from.
            new_power (float): The target power in MeV/cm

        Returns:
            (float) Eigenvalue of the last simulation.

        Todo:
            Provide units for new_power
        """
        import openmc.statepoint as sp

        # Due to some spiteful reason, cell IDs in the Pandas dataframe
        # do not match those specified in the input.
        # It appears, luckily enough, that subtracting 9999 from the
        # cell ID matches the Pandas frame.
        # This will have to be fixed as things go onward.

        magic_shift = 9999

        statepoint = sp.StatePoint(filename)

        k = statepoint.k_combined[0]

        # Generate new power dictionary

        self.power = OrderedDict()

        # ----------------------------------------------------------------------
        # Unpack depletion list
        tally_dep = statepoint.get_tally(id=1)

        df = tally_dep.get_pandas_dataframe()
        for i in range(len(self.burn_list)):
            cell = self.burn_list[i]

            df_cell = df[df["cell"] == cell - magic_shift]

            # For each nuclide that is in the depletion chain
            for key in self.total_number[cell]:
                if key in self.depletion_chain.nuclide_dict:
                    # Check if in participating nuclides
                    if key in self.participating_nuclides:
                        # Pull out nuclide object to iterate over reaction rates
                        nuclide = self.depletion_chain.nuc_by_ind(key)

                        df_nuclide = df_cell[df_cell["nuclide"] == key]

                        for j in range(nuclide.n_reaction_paths):
                            tally_type = nuclide.reaction_type[j]
                            value = df_nuclide[df_nuclide["score"] == tally_type]["mean"].values[0]

                            self.reaction_rates[cell].rate[nuclide.name][j] = value

                            # Calculate power if fission
                            if tally_type == "fission":
                                power = value * nuclide.fission_power
                                if cell not in self.power:
                                    self.power[cell] = power
                                else:
                                    self.power[cell] += power
                    else:
                        # Set reaction rates to zero
                        nuclide = self.depletion_chain.nuc_by_ind(key)

                        df_nuclide = df_cell[df_cell["nuclide"] == key]

                        for j in range(nuclide.n_reaction_paths):
                            self.reaction_rates[cell].rate[nuclide.name][j] = 0.0

        # ----------------------------------------------------------------------
        # Unpack fuel list

        extra_list = list(set(self.fuel_list) - set(self.burn_list))

        # Verify that data exists before reading it.
        if len(extra_list) != 0:
            tally_fuel = statepoint.get_tally(id=2)

            df = tally_fuel.get_pandas_dataframe()

            # We use index 0 chain to get data for these, as to not save chains
            # for each nuclide
            for i in range(len(extra_list)):
                cell = extra_list[i]

                df_cell = df[df["cell"] == cell - magic_shift]

                # For each nuclide that is in the depletion chain
                for key in self.total_number[cell]:
                    if key in self.depletion_chain.nuclide_dict:
                        df_nuclide = df_cell[df_cell["nuclide"] == key]

                        tally_type = "fission"
                        value = df_nuclide[df_nuclide["score"] == tally_type]["mean"].values[0]

                        # Calculate power
                        power = value * nuclide.fission_power
                        if cell not in self.power:
                            self.power[cell] = power
                        else:
                            self.power[cell] += power

        # ----------------------------------------------------------------------
        # Normalize to power
        original_power = sum(self.power.values())

        test = 0.0

        for i in range(len(self.burn_list)):
            cell = self.burn_list[i]

            # For each nuclide that is in the depletion chain
            for key in self.total_number[cell]:
                if key in self.depletion_chain.nuclide_dict:
                    nuclide = self.depletion_chain.nuc_by_ind(key)

                    for j in range(nuclide.n_reaction_paths):

                        if self.number_density[cell][key] != 0:
                            # Normalize reaction rates and divide out number of
                            # nuclides, yielding cross section.
                            self.reaction_rates[cell].rate[nuclide.name][j] *= (new_power / original_power)
                            self.reaction_rates[cell].rate[nuclide.name][j] /= self.total_number[cell][key]

        return k

    def depletion_matrix_list(self):
        """ Generates a list containing the depletion operators.

        Returns a list of sparse (CSR) matrices containing the depletion
        operators for this problem.  It is done in parallel using the concurrent
        futures package.

        Returns:
            (list) A list of scipy.sparse.linalg.csr_matrix.

        Todo:
            Generalize method away from process parallelism.
        """
        import concurrent.futures

        # An issue with concurrent.futures is that it is far easier to write a
        # map, so I need to concatenate the data into a single variable with
        # which a map works.
        input_list = []
        for rate in self.reaction_rates:
            input_list.append((self.depletion_chain, self.reaction_rates[rate]))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            matrices = executor.map(depletion_chain.matrix_wrapper, input_list)

        return list(matrices)

    def total_density_list(self):
        """ Returns a list of total density lists.

        This list is in the exact same order as depletion_matrix_list, so that
        matrix exponentiation can be done easily.

        Returns:
            (list) A list of lists containing number densities of each cell.

        Todo:
            Make this method less fragile.
        """

        total_density = []

        cell_i = 0

        for cell in self.burn_list:

            total_density.append([])

            # Get all nuclides that exist in both chain and total_number
            # in the order of chain
            for i in range(len(self.depletion_chain.nuclides)):
                if self.depletion_chain.nuclides[i].name in self.total_number[cell]:
                    total_density[cell_i].append(self.total_number[cell][self.depletion_chain.nuclides[i].name])
                else:
                    total_density[cell_i].append(0.0)
            cell_i += 1

        return total_density

    def set_density(self, total_density):
        """ Sets density.

        Sets the density in the exact same order as total_density_list outputs,
        allowing for internal consistency

        Args:
            total_density (list): List of lists containing total atoms.

        Todo:
            Make this method less fragile.
        """

        cell_i = 0

        for cell in self.burn_list:

            # Update total_number first
            for i in range(len(self.depletion_chain.nuclides)):
                # Don't add if zero, for performance reasons.
                if total_density[cell_i][i] != 0.0:
                    nuc = self.depletion_chain.nuclides[i].name
                    # Add a "infinitely dilute" quantity if negative
                    # TODO: DEBUG
                    if total_density[cell_i][i] > 0.0:
                        self.total_number[cell][nuc] = total_density[cell_i][i]
                    else:
                        self.total_number[cell][nuc] = 1.0e5

            cell_i += 1

            # Then update number_density
            for nuc in self.total_number[cell]:
                self.number_density[cell][nuc] = self.total_number[cell][nuc] / self.volume[cell]

    def density_dictionary_to_openmc_mat(self, m_id):
        """ Generates an OpenMC material from self.number_density.

        Args:
            m_id (int): Cell ID.

        Returns:
            (openmc.Material) The material filled with nuclides.
        """

        mat = openmc.Material(material_id=m_id)
        total = 0.0
        for key in self.number_density[m_id]:
            nuc = openmc.Nuclide(key)
            mat.add_nuclide(nuc, self.number_density[m_id][key])
            total += self.number_density[m_id][key]
        mat.set_density('atom/cm3', total)

        return mat

    def generate_initial_number_density(self):
        """ Generates initial number density.

        These results were from a CASMO5 run in which the gadolinium pin was
        loaded with 2 wt percent of Gd-157.

        Todo:
            Generalize along with the rest of the geometry for arbitrary
            geometries.
        """

        # Concentration to be used for all fuel pins
        fuel_dict = OrderedDict()
        fuel_dict['U-235'] = 1.05692e21
        fuel_dict['U-234'] = 1.00506e19
        fuel_dict['U-238'] = 2.21371e22
        fuel_dict['O-16'] = 4.62954e22
        fuel_dict['O-17'] = 1.127684e20
        fuel_dict['I-135'] = 1.0e10
        fuel_dict['Xe-135'] = 1.0e10
        fuel_dict['Xe-136'] = 1.0e10
        fuel_dict['Cs-135'] = 1.0e10
        fuel_dict['Gd-156'] = 1.0e10
        fuel_dict['Gd-157'] = 1.0e10
        # fuel_dict['O-18'] = 9.51352e19 # Does not exist in ENDF71, merged into 17

        fuel_gd_dict = OrderedDict()
        fuel_gd_dict['U-235'] = 1.03579e21
        fuel_gd_dict['U-238'] = 2.16943e22
        fuel_gd_dict['Gd-156'] = 3.95517E+10
        fuel_gd_dict['Gd-157'] = 1.08156e20
        fuel_gd_dict['O-16'] = 4.64035e22
        fuel_dict['I-135'] = 1.0e10
        fuel_dict['Xe-136'] = 1.0e10
        fuel_dict['Xe-135'] = 1.0e10
        fuel_dict['Cs-135'] = 1.0e10
        # There are a whole bunch of 1e-10 stuff here.

        # Concentration to be used for cladding
        clad_dict = OrderedDict()
        clad_dict['O-16'] = 3.07427e20
        clad_dict['O-17'] = 7.48868e17
        clad_dict['Cr-50'] = 3.29620e18
        clad_dict['Cr-52'] = 6.35639e19
        clad_dict['Cr-53'] = 7.20763e18
        clad_dict['Cr-54'] = 1.79413e18
        clad_dict['Fe-54'] = 5.57350e18
        clad_dict['Fe-56'] = 8.74921e19
        clad_dict['Fe-57'] = 2.02057e18
        clad_dict['Fe-58'] = 2.68901e17
        clad_dict['Cr-50'] = 3.29620e18
        clad_dict['Cr-52'] = 6.35639e19
        clad_dict['Cr-53'] = 7.20763e18
        clad_dict['Cr-54'] = 1.79413e18
        clad_dict['Ni-58'] = 2.51631e19
        clad_dict['Ni-60'] = 9.69278e18
        clad_dict['Ni-61'] = 4.21338e17
        clad_dict['Ni-62'] = 1.34341e18
        clad_dict['Ni-64'] = 3.43127e17
        clad_dict['Zr-90'] = 2.18320e22
        clad_dict['Zr-91'] = 4.76104e21
        clad_dict['Zr-92'] = 7.27734e21
        clad_dict['Zr-94'] = 7.37494e21
        clad_dict['Zr-96'] = 1.18814e21
        clad_dict['Sn-112'] = 4.67352e18
        clad_dict['Sn-114'] = 3.17992e18
        clad_dict['Sn-115'] = 1.63814e18
        clad_dict['Sn-116'] = 7.00546e19
        clad_dict['Sn-117'] = 3.70027e19
        clad_dict['Sn-118'] = 1.16694e20
        clad_dict['Sn-119'] = 4.13872e19
        clad_dict['Sn-120'] = 1.56973e20
        clad_dict['Sn-122'] = 2.23076e19
        clad_dict['Sn-124'] = 2.78966e19

        # Gap concentration
        # Funny enough, the example problem uses air.
        gap_dict = OrderedDict()
        gap_dict['O-16'] = 7.86548e18
        gap_dict['O-17'] = 2.99548e15
        gap_dict['N-14'] = 3.38646e19
        gap_dict['N-15'] = 1.23717e17

        # Concentration to be used for coolant
        # No boron
        cool_dict = OrderedDict()
        cool_dict['H-1'] = 4.68063e22
        cool_dict['O-16'] = 2.33427e22
        cool_dict['O-17'] = 8.89086e18

        # Store these dictionaries in the initial conditions dictionary
        self.initial_density = OrderedDict()
        self.initial_density['fuel_gd'] = fuel_gd_dict
        self.initial_density['fuel'] = fuel_dict
        self.initial_density['gap'] = gap_dict
        self.initial_density['clad'] = clad_dict
        self.initial_density['cool'] = cool_dict

        # Set up libraries to use
        self.library = OrderedDict()
        self.library_sab = OrderedDict()
        self.sab = OrderedDict()

        MCNP = False

        if MCNP:
            self.library['fuel_gd'] = '82c'
            self.library['fuel'] = '82c'
            # We approximate temperature of everything as 600K, even though it was 580K.
            self.library['gap'] = '81c'
            self.library['clad'] = '81c'
            self.library['cool'] = '81c'

            self.sab['cool'] = 'lwtr'

            self.library_sab['cool'] = '26t'
        else:
            self.library['fuel_gd'] = '71c'
            self.library['fuel'] = '71c'
            self.library['gap'] = '71c'
            self.library['clad'] = '71c'
            self.library['cool'] = '71c'

            self.sab['cool'] = 'HH2O'

            self.library_sab['cool'] = '71t'

    def load_participating(self, filename):
        """ Loads a cross_sections.xml file to find participating nuclides.

        This allows for nuclides that are important in the decay chain but not
        important neutronically, or have no cross section data.

        Args:
            filename (str): Path to cross_sections.xml
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
