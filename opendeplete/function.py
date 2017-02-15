"""function module.

This module contains the Operator class, which is then passed to an integrator
to run a full depletion simulation.
"""

from .depletion_chain import DepletionChain
from .openmc_wrapper import Geometry


class Operator(object):
    """ The Operator class.

    This class contains everything the integrator needs to know to perform a
    simulation.

    Attributes
    ----------
    geometry : Geometry
        The OpenMC geometry object.
    settings : Settings
        Settings object.
    """

    def __init__(self):
        self.geometry = None
        self.settings = None

    def geometry_fill(self, geometry, volume, materials, settings):
        """ Fill operator with OpenMC components.

        Parameters
        ----------
        geometry : openmc.Geometry
            The OpenMC geometry object.
        volume : OrderedDict of int to float
            Given a material ID, gives the volume of said material.
        materials : Materials
            Materials to be used for this simulation.
        settings : Settings
            Settings object.
        """
        # Form geometry
        self.geometry = Geometry(geometry, volume, materials)
        self.settings = settings

        # Load depletion chain
        self.load_depletion_data(settings.chain_file)

        # Initialize geometry
        self.geometry.initialize()

    @property
    def chain(self):
        """DepletionChain
            The depletion chain from the geometry inside.
        """

        return self.geometry.chain

    @property
    def reaction_rates(self):
        """ReactionRates
            Reaction rates from the geometry inside.
        """

        return self.geometry.reaction_rates

    @property
    def total_number(self):
        """OrderedDict of int to OrderedDict of str to float
            Total atoms for the problem.  Indexed by [cell_id]["nuclide name"].
        """

        return self.geometry.total_number

    @property
    def burn_list(self):
        """list of int
            A list of all cell IDs to be burned.  Used for sorting the simulation.
        """

        return self.geometry.burn_list

    def start(self):
        """ Creates initial files, and returns initial vector.

        Returns
        -------
        list of numpy.array
            Total density for initial conditions.
        """

        return self.geometry.start()

    def eval(self, vec):
        """ Runs a simulation.

        Parameters
        ----------
        vec : list of numpy.array
            Total atoms to be used in function.

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
        self.geometry.chain = DepletionChain()
        self.geometry.chain.xml_read(filename)
        self.geometry.fill_nuclide_list()

    def get_results_info(self):
        """ Returns non-participating nuclides, cell lists, and nuc lists.

        Returns
        -------
        nuc_list : list of str
            A list of all nuclide names. Used for sorting the simulation.
        burn_list : list of int
            A list of all cell IDs to be burned.  Used for sorting the simulation.
        not_participating : dict of str to dict of str to float
            Not participating nuclides, indexed by cell id and nuclide id.

        """

        return self.geometry.nuc_list, \
               self.geometry.burn_list, \
               self.geometry.get_non_participating_nuc()
