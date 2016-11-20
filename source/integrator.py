"""Integrator module.

This module contains the actual time integration component of the depletion
algorithm.  This includes matrix exponents, predictor, predictor-corrector, and
eventually more.
"""

import concurrent.futures
import copy
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from results import write_results


def predictor(operator):
    """ Runs a depletion problem using the predictor algorithm.

    This algorithm uses the beginning-of-timestep reaction rates for the whole
    timestep.  This is a first order algorithm.

    Parameters
    ----------
    operator : function.Operator
        The operator object to simulate on.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)

    # Change directory
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.start()

    current_time = 0.0

    for ind, dt in enumerate(operator.settings.dt_vec):
        # Evaluate function at vec to get mat
        mat, eigvl, rates_1, seed = operator.eval(vec)
        write_results(operator, eigvl, [vec], [rates_1], [1], [seed], current_time, ind)

        # Update vec with the integrator.
        vec = matexp(mat, vec, dt)
        current_time += dt

    # Run final simulation
    mat, eigvl, rates_1, seed = operator.eval(vec)
    write_results(operator, eigvl, [vec], [rates_1], [1], [seed], current_time,
                  len(operator.settings.dt_vec))

    # Return to origin
    os.chdir(dir_home)


def ce_cm(operator):
    """ Runs a depletion problem using the center-extrapolate / center-midpoint
    predictor-corrector algorithm.

    This algorithm is a second order algorithm where the predictor algorithm is
    used to get the midpoint number densities, another function is then run,
    and the midpoint reaction rates used for the entire timestep.

    Considered a "constant flux" algorithm.

    Parameters
    ----------
    operator : function.Operator
        The operator object to simulate on.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)

    # Change directory
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.start()

    current_time = 0.0

    for ind, dt in enumerate(operator.settings.dt_vec):
        # Evaluate function at vec to get mat
        mat, eigvl, rates_1, seed_1 = operator.eval(vec)

        # Step a half timestep
        vec_1 = matexp(mat, vec, dt/2)

        # Update function using new RNG
        mat, dummy, rates_2, seed_2 = operator.eval(vec_1)

        write_results(operator, eigvl, [vec, vec_1], [rates_1, rates_2],
                      [0, 1], [seed_1, seed_2], current_time, ind)

        # Step a full timestep
        vec = matexp(mat, vec, dt)
        current_time += dt

    # Run final simulation
    mat, eigvl, rates_1, seed_1 = operator.eval(vec)
    write_results(operator, eigvl, [vec], [rates_1], [1], [seed_1], current_time,
                  len(operator.settings.dt_vec))

    # Return to origin
    os.chdir(dir_home)


def quadratic(operator):
    """ Runs a depletion problem using a quadratic algorithm.

    This algorithm is a third order algorithm in which a linear extrapolation
    of reaction rates from the previous timestep are used for prediction, and a
    quadratic polynomial using previous, current, and predictor reaction rates
    is used to get the final answer.

    From
    ----
        Isotalo, A. E., and P. A. Aarnio. "Higher order methods for burnup
        calculations with Bateman solutions." Annals of Nuclear Energy 38.9
        (2011): 1987-1995.

    Parameters
    ----------
    operator : function.Operator
        The operator object to simulate on.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)

    # Change directory
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec_bos = operator.start()

    cells = len(vec_bos)

    current_time = 0.0

    for ind, dt in enumerate(operator.settings.dt_vec):
        if ind == 0:
            # Constant Extrapolation
            mat_bos, eigvl, rates_1, seed_1 = operator.eval(vec_bos)

            # Get EOS
            vec_eos = matexp(mat_bos, vec_bos, dt)

            # Linear Interpolation
            mat_eos, eigvl, rates_2, seed_2 = operator.eval(vec_eos)

            write_results(operator, eigvl, [vec_bos, vec_eos], [rates_1, rates_2],
                          [1/2, 1/2], [seed_1, seed_2], current_time, ind)

            mat_ext = [0.5*(mat_bos[i] + mat_eos[i])
                       for i in range(cells)]
            vec_bos = matexp(mat_ext, vec_bos, dt)

            # Set mat_prev
            mat_prev = copy.deepcopy(mat_bos)
            rates_prev = copy.deepcopy(rates_1)

            current_time += dt
        else:
            # Constant Extrapolation
            dt_l = operator.settings.dt_vec[i-1]
            mat_bos, eigvl, rates_1, seed_1 = operator.eval(vec_bos)

            # Get EOS
            c1 = (-dt/(2.0 * dt_l))
            c2 = (1 + dt/(2.0 * dt_l))
            mat_int = [mat_prev[i]*c1 + mat_bos[i]*c2 for i in range(cells)]
            vec_eos = matexp(mat_int, vec_bos, dt)

            # Quadratic Extrapolation
            mat_eos, eigvl, rates_2, seed_2 = operator.eval(vec_eos)

            # Store results
            c1 = (-dt**2/(6.0*dt_l*(dt + dt_l)))
            c2 = (0.5 + dt/(6.0*dt_l))
            c3 = (0.5 - dt/(6.0*(dt+dt_l)))
            # TODO find an acceptable compromise for AB-AM schemes.
            write_results(operator, eigvl, [vec_bos, vec_eos], [rates_prev, rates_1, rates_2],
                          [c1, c2, c3], [seed_1, seed_2], current_time, ind)

            # Get new BOS
            mat_ext = [mat_prev[i]*c1 + mat_bos[i]*c2 + mat_eos[i]*c3
                       for i in range(cells)]
            vec_bos = matexp(mat_ext, vec_bos, dt)

            # Set mat_prev
            mat_prev = copy.deepcopy(mat_bos)
            rates_prev = copy.deepcopy(rates_1)

            current_time += dt

    # Run final simulation
    mat_bos, eigvl, rates_1, seed_1 = operator.eval(vec_bos)
    write_results(operator, eigvl, [vec_bos], [rates_1], [1], [seed_1], current_time,
                  len(operator.settings.dt_vec))

    # Return to origin
    os.chdir(dir_home)


def matexp(mat, vec, dt):
    """ Parallel matrix exponent for a list of mat, vec.

    Performs a series of result = exp(mat) * vec in parallel to reduce
    computational load.

    Parameters
    ----------
    mat : List[scipy.sparse.csr_matrix]
        Matrices to take exponent of.
    vec : List[numpy.array]
        Vectors to operate a matrix exponent on.
    dt : float
        Time to integrate to.

    Returns
    -------
    List[numpy.array]
        List of results of the matrix exponent.
    """

    # Merge mat, vec into a list (yes, this is wasteful, blame concurrent
    # having only a map instead of a starmap)

    t1 = time.time()
    data = [(mat[i], vec[i], dt) for i in range(len(mat))]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        vec2 = executor.map(matexp_wrapper, data)
    t2 = time.time()

    print("Time to matexp: ", t2-t1)

    return list(vec2)


def matexp_wrapper(data):
    """ Matexp wrapper for map function.

    Wraps the matrix exponent so that a map can be applied.  Uses CRAM48
    instead of a lower order CRAM method as CRAM already takes a fraction of
    the runtime so using as high order as has been developed costs us nothing.

    Parameters
    ----------
    data : Tuple
        First entry is a csr_matrix, second is the vector, third is the time to
        step to.

    Returns
    -------
    List[numpy.array]
        List of results of the matrix exponent.
    """
    return CRAM48(data[0], np.array(data[1]), data[2])


def CRAM16(A, n0, dt):
    """ Chebyshev Rational Approximation Method, order 16

    Algorithm is the 16th order Chebyshev Rational Approximation Method,
    implemented in the more stable incomplete partial fraction (IPF) form.

    From
    ----
        Pusa, Maria. "Higher-Order Chebyshev Rational Approximation Method and
        Application to Burnup Equations." Nuclear Science and Engineering 182.3
        (2016).

    Parameters
    ----------
    A : scipy.linalg.csr_matrix
        Matrix to take exponent of.
    n0 : numpy.array
        Vector to operate a matrix exponent on.
    dt : float
        Time to integrate to.

    Returns
    -------
    np.array
        Results of the matrix exponent.
    """

    alpha = np.array([+2.124853710495224e-16,
                      +5.464930576870210e+3 - 3.797983575308356e+4j,
                      +9.045112476907548e+1 - 1.115537522430261e+3j,
                      +2.344818070467641e+2 - 4.228020157070496e+2j,
                      +9.453304067358312e+1 - 2.951294291446048e+2j,
                      +7.283792954673409e+2 - 1.205646080220011e+5j,
                      +3.648229059594851e+1 - 1.155509621409682e+2j,
                      +2.547321630156819e+1 - 2.639500283021502e+1j,
                      +2.394538338734709e+1 - 5.650522971778156e+0j],
                     dtype=np.complex128)
    theta = np.array([+0.0,
                      +3.509103608414918 + 8.436198985884374j,
                      +5.948152268951177 + 3.587457362018322j,
                      -5.264971343442647 + 16.22022147316793j,
                      +1.419375897185666 + 10.92536348449672j,
                      +6.416177699099435 + 1.194122393370139j,
                      +4.993174737717997 + 5.996881713603942j,
                      -1.413928462488886 + 13.49772569889275j,
                      -10.84391707869699 + 19.27744616718165j],
                     dtype=np.complex128)

    n = A.shape[0]

    alpha0 = 2.124853710495224e-16

    k = 8

    y = np.array(n0, dtype=np.float64)
    for l in range(1, k+1):
        y = 2.0*np.real(alpha[l]*sla.spsolve(A*dt - theta[l]*sp.eye(n), y)) + y

    y *= alpha0
    return y


def CRAM48(A, n0, dt):
    """ Chebyshev Rational Approximation Method, order 48

    Algorithm is the 48th order Chebyshev Rational Approximation Method,
    implemented in the more stable incomplete partial fraction (IPF) form.

    From
    ----
        Pusa, Maria. "Higher-Order Chebyshev Rational Approximation Method and
        Application to Burnup Equations." Nuclear Science and Engineering 182.3
        (2016).

    Parameters
    ----------
    A : scipy.linalg.csr_matrix
        Matrix to take exponent of.
    n0 : numpy.array
        Vector to operate a matrix exponent on.
    dt : float
        Time to integrate to.

    Returns
    -------
    np.array
        Results of the matrix exponent.
    """

    theta_r = np.array([-4.465731934165702e+1, -5.284616241568964e+0,
                        -8.867715667624458e+0, +3.493013124279215e+0,
                        +1.564102508858634e+1, +1.742097597385893e+1,
                        -2.834466755180654e+1, +1.661569367939544e+1,
                        +8.011836167974721e+0, -2.056267541998229e+0,
                        +1.449208170441839e+1, +1.853807176907916e+1,
                        +9.932562704505182e+0, -2.244223871767187e+1,
                        +8.590014121680897e-1, -1.286192925744479e+1,
                        +1.164596909542055e+1, +1.806076684783089e+1,
                        +5.870672154659249e+0, -3.542938819659747e+1,
                        +1.901323489060250e+1, +1.885508331552577e+1,
                        -1.734689708174982e+1, +1.316284237125190e+1])
    theta_i = np.array([+6.233225190695437e+1, +4.057499381311059e+1,
                        +4.325515754166724e+1, +3.281615453173585e+1,
                        +1.558061616372237e+1, +1.076629305714420e+1,
                        +5.492841024648724e+1, +1.316994930024688e+1,
                        +2.780232111309410e+1, +3.794824788914354e+1,
                        +1.799988210051809e+1, +5.974332563100539e+0,
                        +2.532823409972962e+1, +5.179633600312162e+1,
                        +3.536456194294350e+1, +4.600304902833652e+1,
                        +2.287153304140217e+1, +8.368200580099821e+0,
                        +3.029700159040121e+1, +5.834381701800013e+1,
                        +1.194282058271408e+0, +3.583428564427879e+0,
                        +4.883941101108207e+1, +2.042951874827759e+1])
    theta = np.array(theta_r + theta_i * 1j, dtype=np.complex128)

    alpha_r = np.array([+6.387380733878774e+2, +1.909896179065730e+2,
                        +4.236195226571914e+2, +4.645770595258726e+2,
                        +7.765163276752433e+2, +1.907115136768522e+3,
                        +2.909892685603256e+3, +1.944772206620450e+2,
                        +1.382799786972332e+5, +5.628442079602433e+3,
                        +2.151681283794220e+2, +1.324720240514420e+3,
                        +1.617548476343347e+4, +1.112729040439685e+2,
                        +1.074624783191125e+2, +8.835727765158191e+1,
                        +9.354078136054179e+1, +9.418142823531573e+1,
                        +1.040012390717851e+2, +6.861882624343235e+1,
                        +8.766654491283722e+1, +1.056007619389650e+2,
                        +7.738987569039419e+1, +1.041366366475571e+2])
    alpha_i = np.array([-6.743912502859256e+2, -3.973203432721332e+2,
                        -2.041233768918671e+3, -1.652917287299683e+3,
                        -1.783617639907328e+4, -5.887068595142284e+4,
                        -9.953255345514560e+3, -1.427131226068449e+3,
                        -3.256885197214938e+6, -2.924284515884309e+4,
                        -1.121774011188224e+3, -6.370088443140973e+4,
                        -1.008798413156542e+6, -8.837109731680418e+1,
                        -1.457246116408180e+2, -6.388286188419360e+1,
                        -2.195424319460237e+2, -6.719055740098035e+2,
                        -1.693747595553868e+2, -1.177598523430493e+1,
                        -4.596464999363902e+3, -1.738294585524067e+3,
                        -4.311715386228984e+1, -2.777743732451969e+2])
    alpha = np.array(alpha_r + alpha_i * 1j, dtype=np.complex128)
    n = A.shape[0]

    alpha0 = 2.258038182743983e-47

    k = 24

    y = np.array(n0, dtype=np.float64)
    for l in range(k):
        y = 2.0*np.real(alpha[l]*sla.spsolve(A*dt - theta[l]*sp.eye(n), y)) + y

    y *= alpha0
    return y
