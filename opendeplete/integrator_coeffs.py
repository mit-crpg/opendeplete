"""Integration algorithms module

Contains the coefficients for use in integrator.
"""

import numpy as np

class Integrator(object):
    """ Integrator coefficients class.

    Attributes
    ----------
    a : numpy.ndarray
        The 'a' coefficients for the exponential-linear method.
    d : numpy.ndarray
        The 'd' coefficients for the exponential-linear method.
    p : numpy.ndarray
        The coefficients necessary for constructing an interpolating
        polynomial.
    final_ind : int
        Index of final stage.
    ats_ind : int
        Index of adaptive timestepping stage.
    fsal : bool
        Does this method leverage first-same-as-last?
    ats : bool
        Does this method leverage adaptive timestepping?
    order : int
        The global order of the method.
    p_terms : int
        Number of terms in interpolating polynomial.
    int_terms : int
        Number of terms to use in interpolating polynomial.
    """

    def __init__(self):
        self.a = None
        self.d = None

        self.p = None

        self.final_ind = None
        self.ats_ind = None

        self.fsal = False
        self.ats = False
        self.order = None
        self.p_terms = None
        self.int_terms = None

    @property
    def stages(self):
        """Number of stages in the method."""
        return len(self.d)

# Predictor algorithm with linear-linear interpolation
predictor_c0 = Integrator()
predictor_c0.a = np.zeros((1, 1, 1))
predictor_c0.a[0, 0, 0] = 1
predictor_c0.d = np.zeros((1, 1))
predictor_c0.d[0, 0] = 1
predictor_c0.p = np.zeros((2, 2))
# y0 terms
predictor_c0.p[0, 0] = 1
predictor_c0.p[1, 0] = -1
# y1 terms
predictor_c0.p[0, 1] = 0
predictor_c0.p[1, 1] = 1
predictor_c0.final_ind = 1
predictor_c0.p_terms = 2
predictor_c0.int_terms = 2

# Predictor algorithm with C1 continuous interpolation
predictor_c1 = Integrator()
predictor_c1.a = np.zeros((1, 1, 1))
predictor_c1.a[0, 0, 0] = 1
predictor_c1.d = np.zeros((2, 2))
predictor_c1.d[0, 0] = 1
predictor_c1.p = np.zeros((4, 5))
# y0 terms
predictor_c1.p[0, 0] = 1
predictor_c1.p[1, 0] = 0
predictor_c1.p[2, 0] = -3
predictor_c1.p[3, 0] = 2
# y1 terms
predictor_c1.p[0, 1] = 0
predictor_c1.p[1, 1] = 0
predictor_c1.p[2, 1] = 3
predictor_c1.p[3, 1] = -2
# Skip intermediate stage
# f(y0)y0 terms
predictor_c1.p[0, 3] = 0
predictor_c1.p[1, 3] = 1
predictor_c1.p[2, 3] = -2
predictor_c1.p[3, 3] = 1
# f(y1)y1 terms
predictor_c1.p[0, 4] = 0
predictor_c1.p[1, 4] = 0
predictor_c1.p[2, 4] = -1
predictor_c1.p[3, 4] = 1
predictor_c1.fsal = True
predictor_c1.final_ind = 1
predictor_c1.p_terms = 4
predictor_c1.int_terms = 5

# CE-CM Predictor corrector algorithm with C1 continuous interpolation
ce_cm_c1 = Integrator()
ce_cm_c1.a = np.zeros((2, 2, 2))
ce_cm_c1.a[0, 0, 0] = 1/2
ce_cm_c1.a[1, 0, 1] = 1
ce_cm_c1.d = np.zeros((3, 3))
ce_cm_c1.d[0, 0] = 1
ce_cm_c1.d[1, 0] = 1
# y0 terms
ce_cm_c1.p = np.zeros((4, 7))
ce_cm_c1.p[0, 0] = 1
ce_cm_c1.p[1, 0] = 0
ce_cm_c1.p[2, 0] = -3
ce_cm_c1.p[3, 0] = 2
# Skip intermediate stage
# y1 terms
ce_cm_c1.p[0, 2] = 0
ce_cm_c1.p[1, 2] = 0
ce_cm_c1.p[2, 2] = 3
ce_cm_c1.p[3, 2] = -2
# Skip intermediate stage
# f(y0)y0 terms
ce_cm_c1.p[0, 4] = 0
ce_cm_c1.p[1, 4] = 1
ce_cm_c1.p[2, 4] = -2
ce_cm_c1.p[3, 4] = 1
# Skip derivative of intermediate
# f(y1)y1 terms
ce_cm_c1.p[0, 6] = 0
ce_cm_c1.p[1, 6] = 0
ce_cm_c1.p[2, 6] = -1
ce_cm_c1.p[3, 6] = 1
ce_cm_c1.fsal = True
ce_cm_c1.final_ind = 2
ce_cm_c1.p_terms = 4
ce_cm_c1.int_terms = 7
