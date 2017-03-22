"""Nuclide module.

Contains the per-nuclide components of a depletion chain.
"""

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Nuclide(object):
    """ The Nuclide class.

    Contains everything in a depletion chain relating to a single nuclide.

    Attributes
    ----------
    name : str
        Name of nuclide.
    half_life : float
        Half life of nuclide in s^-1.
    decay_energy : float
        Energy deposited from decay in eV.
    n_decay_paths : int
        Number of decay pathways.
    decay_target : list of str
        Names of targets nuclide can decay to.
    decay_type : list of str
        Name of each decay mode
    branching_ratio : list of float
        Branching ratio for each target.
    n_reaction_paths : int
        Number of possible reaction pathways.
    reaction_target : list of str
        List of names of targets of reactions.
    reaction_type : list of str
        List of names of reactions.
    reaction_Q : list of float
        List of Q values in eV of reactions.
    yield_data : dict of float to list
        Maps tabulated energy to list of (product, yield) for all
        neutron-induced fission products.
    yield_energies : list of float
        Energies at which fission product yiels exist

    """

    def __init__(self):
        # Information about the nuclide
        self.name = None
        self.half_life = None
        self.decay_energy = 0.0

        # Decay paths
        self.decay_target = []
        self.decay_type = []
        self.branching_ratio = []

        # Reaction paths and rates
        self.reaction_target = []
        self.reaction_type = []
        self.reaction_Q = []

        # Neutron fission yields, if present
        self.yield_data = {}
        self.yield_energies = []

    @property
    def n_decay_paths(self):
        """Number of decay pathways."""
        return len(self.decay_target)

    @property
    def n_reaction_paths(self):
        """Number of possible reaction pathways."""
        return len(self.reaction_target)

    @classmethod
    def xml_read(cls, element):
        """Read nuclide from an XML element.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            XML element to write nuclide data to

        Returns
        -------
        nuc : Nuclide
            Instance of a nuclide

        """
        nuc = cls()
        nuc.name = element.get('name')

        # Check for half-life
        if 'half_life' in element.attrib:
            nuc.half_life = float(element.get('half_life'))
            nuc.decay_energy = float(element.get('decay_energy', '0'))

        # Check for decay paths
        for decay_elem in element.iter('decay_type'):
            nuc.decay_target.append(decay_elem.get('target'))
            nuc.decay_type.append(decay_elem.get('type'))
            nuc.branching_ratio.append(
                float(decay_elem.get('branching_ratio')))

        # Check for reaction paths
        for reaction_elem in element.iter('reaction_type'):
            r_type = reaction_elem.get('type')
            nuc.reaction_type.append(r_type)
            nuc.reaction_Q.append(float(reaction_elem.get('Q', '0')))

            # If the type is not fission, get target and Q value, otherwise
            # just set null values
            if r_type != 'fission':
                nuc.reaction_target.append(reaction_elem.get('target'))
            else:
                nuc.reaction_target.append(None)


        fpy_elem = element.find('neutron_fission_yields')
        if fpy_elem is not None:
            for yields_elem in fpy_elem.iter('fission_yields'):
                E = float(yields_elem.get('energy'))
                products = yields_elem.find('products').text.split()
                yields = [float(y) for y in
                          yields_elem.find('data').text.split()]
                nuc.yield_data[E] = list(zip(products, yields))
            nuc.yield_energies = list(sorted(nuc.yield_data.keys()))

        return nuc

    def xml_write(self):
        """Write nuclide to XML element.

        Returns
        -------
        elem : xml.etree.ElementTree.Element
            XML element to write nuclide data to

        """
        elem = ET.Element('nuclide_table')
        elem.set('name', self.name)

        if self.half_life is not None:
            elem.set('half_life', str(self.half_life))
            elem.set('decay_modes', str(len(self.decay_type)))
            elem.set('decay_energy', str(self.decay_energy))
            for mode, daughter, br in zip(self.decay_type, self.decay_target,
                                          self.branching_ratio):
                mode_elem = ET.SubElement(elem, 'decay_type')
                mode_elem.set('type', mode)
                mode_elem.set('target', daughter)
                mode_elem.set('branching_ratio', str(br))

        elem.set('reactions', str(len(self.reaction_type)))
        for rx, daughter, Q in zip(self.reaction_type, self.reaction_target,
                                   self.reaction_Q):
            rx_elem = ET.SubElement(elem, 'reaction_type')
            rx_elem.set('type', rx)
            rx_elem.set('Q', str(Q))
            if rx != 'fission':
                rx_elem.set('target', daughter)

        if self.yield_data:
            fpy_elem = ET.SubElement(elem, 'neutron_fission_yields')
            energy_elem = ET.SubElement(fpy_elem, 'energies')
            energy_elem.text = ' '.join(str(E) for E in self.yield_energies)

            for E in self.yield_energies:
                yields_elem = ET.SubElement(fpy_elem, 'fission_yields')
                yields_elem.set('energy', str(E))

                products_elem = ET.SubElement(yields_elem, 'products')
                products_elem.text = ' '.join(x[0] for x in self.yield_data[E])
                data_elem = ET.SubElement(yields_elem, 'data')
                data_elem.text = ' '.join(str(x[1]) for x in self.yield_data[E])

        return elem
