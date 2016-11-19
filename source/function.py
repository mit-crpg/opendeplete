"""function module.

This module contains the Operator class, which is then passed to an integrator
to run a full depletion simulation.
"""

import depletion_chain
import openmc_wrapper


class Operator:
    """ The Operator class.

    This class contains everything the integrator needs to know to perform a
    simulation.

    Parameters
    ----------
    geometry : openmc.Geometry
        The OpenMC geometry object.
    volume : OrderedDict[float]
        Given a material ID, gives the volume of said material.
    materials : openmc_wrapper.Materials
        Materials to be used for this simulation.
    settings : openmc_wrapper.Settings
        Settings object.

    Attributes
    ----------
    geometry : openmc_wrapper.Geometry
        The OpenMC geometry object.
    settings : openmc_wrapper.Settings
        Settings object.
    """

    def __init__(self, geometry, volume, materials, settings):
        # Form geometry
        self.geometry = openmc_wrapper.Geometry(geometry, volume, materials)
        self.settings = settings

        # Load depletion chain
        self.load_depletion_data(settings.chain_file)

        # Initialize geometry
        self.geometry.initialize()

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
