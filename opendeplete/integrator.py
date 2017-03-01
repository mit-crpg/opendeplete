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

from .results import Results, write_results

def integrate(operator, coeffs, print_out=True):
    """ Performs integration of an operator using the method in coeffs.

    This is a general exponential-linear type integrator for depletion.
    It supports adaptive timestepping, first-same-as-last, and interpolation.

    Parameters
    ----------
    operator : Operator
        The operator object to simulate on.
    coeffs : Integrator
        Coefficients to use to integrate with.
    print_out : bool, optional
        Whether or not to print out time.
    """

    # Save current directory
    dir_home = os.getcwd()

    # Move to folder
    os.makedirs(operator.settings.output_dir, exist_ok=True)

    # Change directory
    os.chdir(operator.settings.output_dir)

    # Generate initial conditions
    vec = operator.initial_condition()

    # Initial configuration
    current_time = 0.0
    end_time = np.sum(operator.settings.dt_vec)
    step_ind = 1

    # Compute initial time step and what stage we should pause on to check
    # errors before interpolation
    if operator.settings.tol is not None and coeffs.ats:
        dt = 3600 # 1 hour
        int_step = coeffs.ats_ind
        ats_mode = True
        tol = operator.settings.tol
    else:
        dt = operator.settings.dt_vec[0]
        int_step = coeffs.final_ind
        ats_mode = False

    # Storage for fsal capabilities
    mat_last = []
    rates_last = []
    eigvl_last = 0
    seed_last = 0

    cells = len(vec)

    while current_time < end_time:
        # Create vectors
        x = [copy.copy(vec)]
        f = []
        seeds = []
        eigvls = []
        rates_array = []

        # Ensure timestep does not fall off edge
        if current_time + dt > end_time:
            dt = end_time - current_time
            # Ensure termination in case of roundoff
            if dt < 1.0e-12 * end_time:
                break

        # For each component vector
        for s in range(int_step):
            # Compute f as needed
            if step_ind == 1 or not coeffs.fsal or s > 0:
                mat, eigvl, rates, seed = operator.eval(x[s])
            else:
                # Use stored FSAL data
                mat = mat_last
                eigvl = eigvl_last
                rates = rates_last
                seed = seed_last

            # Store values
            f.append(mat)
            eigvls.append(eigvl)
            seeds.append(seed)
            rates_array.append(copy.deepcopy(rates))

            # Compute next x
            x_next = compute_x(coeffs, f, x, dt, s, print_out)
            x.append(x_next)

        # Compute error if needed
        if ats_mode:
            relerr = compute_max_relerr(x[coeffs.final_ind], x[coeffs.ats_ind])

            if relerr > tol:
                # Compute new timestep
                dt = 0.9 * dt * (tol / relerr)**(1 / coeffs.order)
                continue

        # Compute rest of stages (for interpolation)
        for s in range(int_step, coeffs.stages):
            # Compute f
            mat, eigvl, rates, seed = operator.eval(x[s])

            # Store values
            f.append(mat)
            eigvls.append(eigvl)
            seeds.append(seed)
            rates_array.append(copy.deepcopy(rates))

            # Compute next x
            x_next = compute_x(coeffs, f, x, dt, s)
            x.append(x_next)

        # Compute point-derivatives of x where known and append
        for i in range(len(f)):
            dx = [dt * (f[i][j] * x[i][j]) for j in range(cells)]
            x.append(dx)

        # Compute interpolating polynomials, fill in results
        results = compute_results(operator, coeffs, x)

        results.final_stage = coeffs.final_ind
        results.k = eigvls
        results.seeds = seeds
        results.time = [current_time, current_time + dt]
        results.rates = rates_array

        write_results(results, "results", step_ind)

        # Compute next time step and store values if FSAL
        current_time += dt
        step_ind += 1
        vec = copy.deepcopy(x[coeffs.final_ind])

        if ats_mode:
            dt = 0.9 * dt * (tol / relerr)**(1 / coeffs.order)

        if coeffs.fsal:
            mat_last = copy.deepcopy(f[coeffs.final_ind])
            eigvl_last = copy.deepcopy(eigvls[coeffs.final_ind])
            rates_last = copy.deepcopy(rates_array[coeffs.final_ind])
            seed_last = copy.deepcopy(seeds[coeffs.final_ind])

    # Return to origin
    os.chdir(dir_home)

def compute_x(coeffs, f, x, dt, stage, print_out=True):
    r""" Compute sub-vectors x for exponential-linear

    This function computes x using the following equation
    .. math:
        x_s = \sum_{i=1}^s d_{si} e^{h \sum_{j=1}^s a_{sij} F(x_j)} x_i

    Parameters
    ----------
    coeffs : Integrator
        Coefficients to use in this calculation.
    f : list of list of scipy.sparse.csr_matrix
        The depletion matrices.  Indexed [j][cell] using the above equation.
    x : list of list of numpy.array
        The prior x vectors.  Indexed [i][cell] using the above equation.
    dt : Float
        The current timestep.
    stage : Int
        Index s in the above equation
    print_out : bool, optional
        Whether or not to print out time.

    Returns
    -------
    list of numpy.array
        The next x component for each cell.
    """

    # List for sub-vectors
    x_sub = []

    cells = len(x[0])

    for i in range(stage + 1):
        # To save some time, determine if this index i is needed
        if coeffs.d[stage, i] == 0:
            continue

        # Compute matrix sum in exponent
        mat = []

        for mat_ind in range(cells):
            mat_cell = coeffs.a[stage, i, 0] * f[0][mat_ind]
            for j in range(1, stage + 1):
                mat_cell += coeffs.a[stage, i, j] * f[j][mat_ind]
            mat.append(mat_cell)

        matrix_exponent = matexp(mat, x[i], dt, print_out=print_out)
        x_a = [coeffs.d[stage, i] * matrix_exponent[j] for j in range(cells)]

        x_sub.append(x_a)

    return vector_sum(x_sub)

def compute_max_relerr(v1, v2):
    """ Compute the maximum relative error between v1 and v2.

    relerr = abs((v1 - v2)/v1)

    Parameters
    ----------
    v1 : list of numpy.array
        Vector 1 for each cell.
    v2 : list of numpy.array
        Vector 2 for each cell.

    Returns
    -------
    relerr : Float
        Relative error.
    """

    relerr = 0.0

    cells = len(v1)

    for i in range(cells):
        relerr_cell = max(np.abs((v1[i] - v2[i]) / v1[i]))
        if relerr_cell > relerr:
            relerr = relerr_cell
    return relerr

def compute_results(op, coeffs, x):
    r""" Computes polynomial coefficients and stores into a results type

    Computes the polynomial coefficients given by
    .. math:
        c_i = \sum_{j=1}^n p_{ij} x_{j}
    where n is the number of components of x

    Parameters
    ----------
    op : Function
        The operator used to generate these results.
    coeffs : Integrator
        Coefficients to use in this calculation.
    x : list of list of numpy.array
        The prior x vectors.  Indexed [i][cell] using the above equation.

    Returns
    -------
    results : Results
        A mostly-filled results object for saving to disk.
    """

    # Get indexing terms
    vol_list, nuc_list, burn_list = op.get_results_info()

    # Create results
    results = Results()
    results.allocate(vol_list, nuc_list, burn_list, coeffs.p_terms)

    for mat_i, mat in enumerate(burn_list):
        mat_str = str(mat)
        # Compute polynomials
        n_nuc = len(x[0][mat_i])

        p = np.zeros((n_nuc, coeffs.p_terms))

        for i in range(coeffs.p_terms):
            for j in range(coeffs.int_terms):
                if coeffs.p[i, j] != 0:
                    p[:, i] += coeffs.p[i, j] * x[j][mat_i]

        # Add to results
        for nuc_i, nuc in enumerate(nuc_list):
            results[mat_str, nuc] = p[nuc_i, :]

    return results

def vector_sum(vecs):
    """ Computes the sum of a list of vectors.

    Parameters
    ----------
    vecs : list of list of numpy.array
        A list of list of arrays.  The internal list of arrays will be summed.

    Returns
    -------
    vec : list of numpy.array
        The summed components.
    """
    if vecs == []:
        return []
    vec = copy.copy(vecs[0])

    for i in range(1, len(vecs)):
        for j in range(len(vecs[i])):
            vec[j] += vecs[i][j]

    return vec

def matexp(mat, vec, dt, print_out=True):
    """ Parallel matrix exponent for a list of mat, vec.

    Performs a series of result = exp(mat) * vec in parallel to reduce
    computational load.

    Parameters
    ----------
    mat : list of scipy.sparse.csr_matrix
        Matrices to take exponent of.
    vec : list of numpy.array
        Vectors to operate a matrix exponent on.
    dt : float
        Time to integrate to.
    print_out : bool, optional
        Whether or not to print out time.

    Returns
    -------
    list of numpy.array
        List of results of the matrix exponent.
    """

    t1 = time.time()

    def data_iterator(start, end):
        """ Simple iterator over matrices and vectors."""
        i = start

        while i < end:
            yield (mat[i], vec[i], dt)
            i += 1

    with concurrent.futures.ProcessPoolExecutor() as executor:
        vec2 = executor.map(matexp_wrapper, data_iterator(0, len(mat)))
    t2 = time.time()

    if print_out:
        print("Time to matexp: ", t2-t1)

    return list(vec2)


def matexp_wrapper(data):
    """ Matexp wrapper for map function.

    Wraps the matrix exponent so that a map can be applied.  Uses CRAM48
    instead of a lower order CRAM method as CRAM already takes a fraction of
    the runtime so using as high order as has been developed costs us nothing.

    Parameters
    ----------
    data : tuple of scipy.sparse.csr_matrix, numpy.array, float
        First entry is a csr_matrix, second is the vector, third is the time to
        step to.

    Returns
    -------
    list of numpy.array
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
    numpy.array
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
    numpy.array
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
