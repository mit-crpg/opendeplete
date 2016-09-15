"""An example file showing how to make a geometry.

This particular example creates a 2x2 geometry, with 3 regular pins and one
Gd-157 2 wt-percent enriched.  The geometry is reflected, with reflections
going through the center of the Gd-157 pin.  Said pin is split into 5 rings.
"""

import function
import openmc
import openmc_wrapper
import numpy as np
from collections import OrderedDict


def generate_initial_number_density():
    """ Generates initial number density.

    These results were from a CASMO5 run in which the gadolinium pin was
    loaded with 2 wt percent of Gd-157.
    """

    # Concentration to be used for all fuel pins
    fuel_dict = OrderedDict()
    fuel_dict['U235'] = 1.05692e21
    fuel_dict['U234'] = 1.00506e19
    fuel_dict['U238'] = 2.21371e22
    fuel_dict['O16'] = 4.62954e22
    fuel_dict['O17'] = 1.127684e20
    fuel_dict['I135'] = 1.0e10
    fuel_dict['Xe135'] = 1.0e10
    fuel_dict['Xe136'] = 1.0e10
    fuel_dict['Cs135'] = 1.0e10
    fuel_dict['Gd156'] = 1.0e10
    fuel_dict['Gd157'] = 1.0e10
    # fuel_dict['O18'] = 9.51352e19 # Does not exist in ENDF71, merged into 17

    # Concentration to be used for the gadolinium fuel pin
    fuel_gd_dict = OrderedDict()
    fuel_gd_dict['U235'] = 1.03579e21
    fuel_gd_dict['U238'] = 2.16943e22
    fuel_gd_dict['Gd156'] = 3.95517E+10
    fuel_gd_dict['Gd157'] = 1.08156e20
    fuel_gd_dict['O16'] = 4.64035e22
    fuel_dict['I135'] = 1.0e10
    fuel_dict['Xe136'] = 1.0e10
    fuel_dict['Xe135'] = 1.0e10
    fuel_dict['Cs135'] = 1.0e10
    # There are a whole bunch of 1e-10 stuff here.

    # Concentration to be used for cladding
    clad_dict = OrderedDict()
    clad_dict['O16'] = 3.07427e20
    clad_dict['O17'] = 7.48868e17
    clad_dict['Cr50'] = 3.29620e18
    clad_dict['Cr52'] = 6.35639e19
    clad_dict['Cr53'] = 7.20763e18
    clad_dict['Cr54'] = 1.79413e18
    clad_dict['Fe54'] = 5.57350e18
    clad_dict['Fe56'] = 8.74921e19
    clad_dict['Fe57'] = 2.02057e18
    clad_dict['Fe58'] = 2.68901e17
    clad_dict['Cr50'] = 3.29620e18
    clad_dict['Cr52'] = 6.35639e19
    clad_dict['Cr53'] = 7.20763e18
    clad_dict['Cr54'] = 1.79413e18
    clad_dict['Ni58'] = 2.51631e19
    clad_dict['Ni60'] = 9.69278e18
    clad_dict['Ni61'] = 4.21338e17
    clad_dict['Ni62'] = 1.34341e18
    clad_dict['Ni64'] = 3.43127e17
    clad_dict['Zr90'] = 2.18320e22
    clad_dict['Zr91'] = 4.76104e21
    clad_dict['Zr92'] = 7.27734e21
    clad_dict['Zr94'] = 7.37494e21
    clad_dict['Zr96'] = 1.18814e21
    clad_dict['Sn112'] = 4.67352e18
    clad_dict['Sn114'] = 3.17992e18
    clad_dict['Sn115'] = 1.63814e18
    clad_dict['Sn116'] = 7.00546e19
    clad_dict['Sn117'] = 3.70027e19
    clad_dict['Sn118'] = 1.16694e20
    clad_dict['Sn119'] = 4.13872e19
    clad_dict['Sn120'] = 1.56973e20
    clad_dict['Sn122'] = 2.23076e19
    clad_dict['Sn124'] = 2.78966e19

    # Gap concentration
    # Funny enough, the example problem uses air.
    gap_dict = OrderedDict()
    gap_dict['O16'] = 7.86548e18
    gap_dict['O17'] = 2.99548e15
    gap_dict['N14'] = 3.38646e19
    gap_dict['N15'] = 1.23717e17

    # Concentration to be used for coolant
    # No boron
    cool_dict = OrderedDict()
    cool_dict['H1'] = 4.68063e22
    cool_dict['O16'] = 2.33427e22
    cool_dict['O17'] = 8.89086e18

    # Store these dictionaries in the initial conditions dictionary
    initial_density = OrderedDict()
    initial_density['fuel_gd'] = fuel_gd_dict
    initial_density['fuel'] = fuel_dict
    initial_density['gap'] = gap_dict
    initial_density['clad'] = clad_dict
    initial_density['cool'] = cool_dict

    # Set up libraries to use
    temperature = OrderedDict()
    sab = OrderedDict()

    # Toggle betweeen MCNP and NNDC data
    MCNP = False

    if MCNP:
        temperature['fuel_gd'] = 900.0
        temperature['fuel'] = 900.0
        # We approximate temperature of everything as 600K, even though it was
        # actually 580K.
        temperature['gap'] = 600.0
        temperature['clad'] = 600.0
        temperature['cool'] = 600.0
    else:
        temperature['fuel_gd'] = 293.6
        temperature['fuel'] = 293.6
        temperature['gap'] = 293.6
        temperature['clad'] = 293.6
        temperature['cool'] = 293.6

    sab['cool'] = 'c_H_in_H2O'

    # Set up burnable materials
    burn = OrderedDict()
    burn['fuel_gd'] = True
    burn['fuel'] = True
    burn['gap'] = False
    burn['clad'] = False
    burn['cool'] = False

    materials = openmc_wrapper.Materials()
    materials.temperature = temperature
    materials.sab = sab
    materials.initial_density = initial_density
    materials.burn = burn

    return materials


def generate_geometry():
    """ Generates example geometry.

    This function creates the initial geometry, a 4 pin reflective problem.
    One pin, containing gadolinium, is discretized into 5 radial cells of
    equal volume.  Reflections go through the center of this pin.

    In addition to what one would do with the general OpenMC geometry code, it
    is necessary to create a dictionary, volume, that maps a cell ID to a
    volume. Further, by naming cells the same as the above materials, the code
    can automatically handle the mapping.
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

    # Form dictionaries for later use.
    volume = OrderedDict()

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
    gd_fuel_r = [openmc.ZCylinder(x0=0, y0=0, R=r_rings[i])
                 for i in range(n_rings)]
    gd_clad_ir = openmc.ZCylinder(x0=0, y0=0, R=r_gap)
    gd_clad_or = openmc.ZCylinder(x0=0, y0=0, R=r_clad)

    # Fuel rings
    gd_fuel_cell = [openmc.Cell(name='fuel_gd') for i in range(n_rings)]

    gd_fuel_cell[0].region = -gd_fuel_r[0] & +left & +bottom
    volume[gd_fuel_cell[0].id] = v_ring / 4.0

    for i in range(n_rings-1):
        gd_fuel_cell[i+1].region = +gd_fuel_r[i] & -gd_fuel_r[i+1] &\
                                   +left & +bottom
        volume[gd_fuel_cell[i+1].id] = v_ring / 4.0

    # Gap
    gd_fuel_gap = openmc.Cell(name='gap')
    gd_fuel_gap.region = +gd_fuel_r[n_rings-1] & -gd_clad_ir & +left & +bottom
    volume[gd_fuel_gap.id] = v_gap / 4.0

    # Clad
    gd_fuel_clad = openmc.Cell(name='clad')
    gd_fuel_clad.region = +gd_clad_ir & -gd_clad_or & +left & +bottom
    volume[gd_fuel_clad.id] = v_clad / 4.0

    # ----------------------------------------------------------------------
    # Fill pin 2, 3 and 4 (without gadolinium)
    coords = [[pitch, 0], [0, pitch], [pitch, pitch]]

    fuel_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_fuel) for x in coords]
    clad_ir_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_gap) for x in coords]
    clad_or_s = [openmc.ZCylinder(x0=x[0], y0=x[1], R=r_clad) for x in coords]

    fuel_cell = [openmc.Cell(name='fuel') for x in coords]
    clad_cell = [openmc.Cell(name='clad') for x in coords]
    gap_cell = [openmc.Cell(name='gap') for x in coords]

    fuel_cell[0].region = -fuel_s[0] & +bottom
    fuel_cell[1].region = -fuel_s[1] & +left
    fuel_cell[2].region = -fuel_s[2]

    gap_cell[0].region = +fuel_s[0] & -clad_ir_s[0] & +bottom
    gap_cell[1].region = +fuel_s[1] & -clad_ir_s[1] & +left
    gap_cell[2].region = +fuel_s[2] & -clad_ir_s[2]

    clad_cell[0].region = +clad_ir_s[0] & -clad_or_s[0] & +bottom
    clad_cell[1].region = +clad_ir_s[1] & -clad_or_s[1] & +left
    clad_cell[2].region = +clad_ir_s[2] & -clad_or_s[2]

    volume[fuel_cell[0].id] = v_fuel / 2.0
    volume[fuel_cell[1].id] = v_fuel / 2.0
    volume[fuel_cell[2].id] = v_fuel

    volume[gap_cell[0].id] = v_gap / 2.0
    volume[gap_cell[1].id] = v_gap / 2.0
    volume[gap_cell[2].id] = v_gap

    volume[clad_cell[0].id] = v_clad / 2.0
    volume[clad_cell[1].id] = v_clad / 2.0
    volume[clad_cell[2].id] = v_clad

    # ----------------------------------------------------------------------
    # Fill coolant

    cool_cell = openmc.Cell(name='cool')
    cool_cell.region = +clad_or_s[0] & +clad_or_s[1] & +clad_or_s[2] &\
                       +gd_clad_or & +left & -right & +bottom & -top
    volume[cool_cell.id] = (3/2 * pitch)**2 - 2.25 * v_fuel - \
        2.25 * v_gap - 2.25 * v_clad

    # ----------------------------------------------------------------------
    # Finalize geometry
    root = openmc.Universe(universe_id=0, name='root universe')

    root.add_cells([cool_cell] + clad_cell + gap_cell + fuel_cell +
                   gd_fuel_cell + [gd_fuel_clad] + [gd_fuel_gap])

    geometry = openmc.Geometry(root)
    return geometry, volume
