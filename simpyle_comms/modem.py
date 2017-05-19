"""
Author: John Sochacki
This module is a collection of matlab like functions that allow me to run
matlab like code and do communications work in python.
"""

import numpy as np
import scipy as sp

class TimeDomainSignal(object):
    """
    This is

    Parameters
    ----------

        None : None
               None?

    Example
    -------

    >>> instance = HRBOMInformation()
    <Sochacki.SochackiUtilityPackage.StringListCollector at 0xab73a58>
    >>> instance.bom_dom.CPN.items
    ['1231231', 'CL-00000031231', 'aggg text!  why??']
    or
    >>> instance = HRBOMInformation({'sheet1':'val1','sheet2':{}})
    """
    def __init__(self, *args):
        pass

    @property
    def document_name(self):
        pass

    @staticmethod
    def convolution_filter(self, filter):
        pass

class Functions(object):
    """
    This is

    Parameters
    ----------

        None : None
               None?

    Example
    -------

    >>> instance = HRBOMInformation()
    <Sochacki.SochackiUtilityPackage.StringListCollector at 0xab73a58>
    >>> instance.bom_dom.CPN.items
    ['1231231', 'CL-00000031231', 'aggg text!  why??']
    or
    >>> instance = HRBOMInformation({'sheet1':'val1','sheet2':{}})
    """
    def __init__(self, *args):
        self.pi = np.pi

    @property
    def document_name(self):
        pass

    @staticmethod
    def circ_shift_2_to_1(x, y):
        pass

    @staticmethod
    def xcorr(x, y):
        if len(x) < len(y):
            x, y = y, x
        return np.correlate(x, y, mode='full')

    @staticmethod
    def linear_convolution(x, y):
        # Theirs is so much faster and I have checked that it provides the
        # exact same answer as cls.linear_convolution(x, y) so just use theirs
        # theirs is np.convolve(x, y)
        y = np.array(y)
        if len(x) < len(y):
            x, y = y, x
        N = len(x)
        M = len(y)
        x = np.array(sum([[0] * (M-1), list(x), [0] * (M-1)], []))
        result = np.zeros(shape=(M + N - 1, ))
        for Index in range(0, M + N - 1):
            result[Index] = sum(y * x[range(Index + M - 1, Index - 1, -1)])
        return result


class Constants(object):
    """
    This is

    Parameters
    ----------

        None : None
               None?

    Example
    -------

    >>> instance = HRBOMInformation()
    <Sochacki.SochackiUtilityPackage.StringListCollector at 0xab73a58>
    >>> instance.bom_dom.CPN.items
    ['1231231', 'CL-00000031231', 'aggg text!  why??']
    or
    >>> instance = HRBOMInformation({'sheet1':'val1','sheet2':{}})
    """
    def __init__(self, *args):
        self.pi = np.pi

# Required imports if put in sepatate package
# from Sochacki.SochackiUtilityPackage import AttrDict
# from Sochacki.SochackiUtilityPackage import StringListCollector