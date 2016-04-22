"""function module.

This module contains the Operator class, which is then passed to an integrator
to run a full depletion simulation.
"""

import depletion_chain
import reaction_rates
import openmc
import openmc_wrapper
from collections import OrderedDict


class Operator:
    """ The Operator class.

    This class contains everything the integrator needs to know to perform a
    simulation.

    Attributes
    ----------
    geometry : openmc_wrapper.Geometry
        The OpenMC geometry object.
    settings : openmc_wrapper.Settings
        Settings object.
    """

    def __init__(self):
        self.geometry = None
        self.settings = None

    @property
    def chain(self):
        """depletion_chain.DepletionChain:
            The depletion chain from the geometry inside."""

        return self.geometry.chain

    @property
    def reaction_rates(self):
        """reaction_rates.ReactionRates:
            Reaction rates from the geometry inside."""

        return self.geometry.reaction_rates

    @property
    def total_number(self):
        """OrderedDict[int : OrderedDict[str : float]]:
            Total atoms for the problem.  Indexed by [cell_id]["nuclide name"].
        """

        return self.geometry.total_number

    def initialize(self, geo, vol, mat, settings):
        """ Initializes the Operator.

        Loads all the necessary data into the object for later passing to an
        integrator function.

        Parameters
        ----------
        geo : openmc.Geometry
            The geometry to simulate.
        vol : dict[float]
            A dictionary mapping volumes to cell IDs.
        mat : openmc_wrapper.Materials
            Materials settings for the problem.
        settings : openmc_wrapper.Settings
            OpenMC simulation settings.
        """

        self.geometry = openmc_wrapper.Geometry()
        # First, load in depletion data
        self.load_depletion_data(settings.chain_file)
        # Then, create geometry
        self.geometry.geometry = geo
        self.geometry.materials = mat
        self.geometry.volume = vol
        self.geometry.initialize(settings)

        # Save settings
        self.settings = settings

    def start(self):
        """ Creates initial files, and returns initial vector.

        Returns
        -------
        List[numpy.array]
            Total density for initial conditions.
        """

        return self.geometry.start()

    def set_density(self, vec):
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

        self.geometry.set_density(vec)

    def eval(self, vec):
        """ Runs a simulation.

        Parameters
        ----------
        vec : list[np.array]
            Total atoms to be used in function.

        Returns
        -------
        mat : list[csr_matrix]
            Matrices for the next step.
        k : float
            Eigenvalue of the problem.
        rates : ReactionRates
            Reaction rates from this simulation.
        seed : int
            Seed for this simulation.
        """

        return self.geometry.function_evaluation(vec, self.settings)

    def load_depletion_data(self, filename):
        """ Load self.depletion_data

        Loads the depletion data from an .xml file.

        Parameters
        ----------
        filename : str
            Filename to load .xml from.
        """

        # Create a depletion chain object, and then allocate the reaction
        # rate objects
        self.geometry.chain = depletion_chain.DepletionChain()
        self.geometry.chain.xml_read(filename)
