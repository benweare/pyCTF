'''
Abstract classes.

This module contains abstract classes used by other classes in 
PyCTF.

'''


class LineProfiles:
    '''
    Abstract class to contain line profiles. 

    Arguments
    ---------
    radial_profile : array 
    frequency : array 
    smoothed_profile : array 
    baseline : array 
    cropped_profile : array
    bins : int

    '''
    def __init__( self ):
        self.radial_profile = None
        self.frequency = None
        self.smoothed_profile = None
        self.baseline = None
        self.cropped_profile = None
        self.cropped_frequency = None
        self.bins = None
        return


class LensAberrations:
    '''
    Abstract class to contain line profiles.

    Arguments 
    ---------
    piston : float
    tilt : float
    defocus : float
    C20 : float
    C12 : float
    Cs : float
    spherical_aberration : float
    C30 : float
    C3 : float

    '''
    # Names are aliased with common names for aberrations.
    def __init__( self ):
        # piston
        self.piston = None
        # tilt
        self.tilt = None
        # defocus
        self.defocus = None
        self.C20 = self.defocus
        # twofold astigmatism
        self.C12 = [None, None]
        self.twofold_astigmatism = self.C12
        # spherical aberration
        self.Cs = None
        self.spherical_aberration = self.Cs
        self.C30 = self.Cs
        self.C3 = self.Cs
        return


class ZerosData:
    '''
    Abstract class to contain minima of CTF. 

    Arguments
    ---------
    maxima : array
    minima : array
    x_min : array
    y_min : array
    indicies_min : array
    indicies_max : array
    results : class

    '''
    def __init__( self ):
        self.maxima = None
        self.minima = None
        self.x_min = None
        self.y_min = None
        self.indicies_min = None
        self.indicies_max = None
        self.results = None
        return