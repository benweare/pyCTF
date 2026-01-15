'''
Module containing miscellanous functions. 

This module contains miscellanous functions used by other  modules 
in the PyCTF package.
'''

import numpy as np

import scipy
from scipy import sparse
from scipy.constants import( e, c, m_e, h )
from scipy.sparse.linalg import spsolve 

def kv_to_lamb( kV ):
    """
    Calculate accelerating voltage from wavelength.

    Parameters
    ----------
    kV : float
        accelerating voltage

    Returns
    -------
    lamb : float
        electron wavelength

    Notes
    -----
    Calulate relativistic electron wavelength from accelerating voltage, 
    using the standard equation (Williams and Carter, (1996)).

    Used by classes: CTF_image, CTF_simulation_1D, CTF_simulation_2D.
    """
    E = kV*1000
    PT = scipy.constants.h * scipy.constants.c
    PBA = (scipy.constants.e *E)*(scipy.constants.e *E)
    PBB = 2*scipy.constants.e*E*scipy.constants.m_e*(scipy.constants.c)\
    *(scipy.constants.c)
    lamb = PT/np.sqrt(PBA+PBB)#lambda in metres
    return lamb

def normalise_data_range( data, **kwargs ):
    '''
    Normalise range of array.

    Parameters
    ----------
    data : array-like
        Input array.

    Returns
    -------
    array-like
        Normalised array.

    Notes
    -----
    Normalise data to a range using feature scaling. 

    '''
    dmin = kwargs.get('dmin', 0)
    dmax = kwargs.get('dmax', 1)
    return ((data-np.min(data))/(np.max(data)-np.min(data)))*( dmax - dmin )

def baseline_als( y, lam, p, niter ):
    """
    Baseline correction for 1D datasets.

    Parameters
    ----------
    y : array_like
        Input data.
    p : float
        Value for asymmetry (0.1 to 0.001).
    lam : float
        Value for smoothing (range 100 to 1000).
    niter : int

    Returns
    -------
    z : numpy array
        Smoothed data.

    Notes
    -----
    Baseline correction via asymmetric least squares smoothing. Based on the 
    method of Eilersand Boelens (2005), via Baek et al. (2014).

    """
    n_iterations = 10
    L = len( y )
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones( L )
    for i in range( n_iterations ):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# (y=m*x+c) for fitting
def gradient_simple( x, m, c ):
    """
    Gradient of a straight line.

    Parameters
    ----------
    x : float
    m : float
    c : float

    Returns
    -------
    y : float

    Notes
    -----
    Used by classes: CTF_image, twofoldAstigmatism, chromaticAberration.
    """
    y = m * x + c
    return y

def composite_image( image1, image2, size ):
    '''
    Make a composite of two images. 

    Parameters
    ----------
    image_1 : array_like
        Image data.
    image_2 : array_like
        Image data.
    size : float
        Size of image.
    niter : int

    Returns
    -------
    composite : array-like
        Composited image.

    Notes
    -----
    Takes two images, and returns a copy of the first image with the lower 
    right quarter replaced with the lower right quarter of the second image. 
    Suitable for images with a 1:1 aspect ratio.

    '''
    composite = np.zeros( (np.size(image1, 0), np.size(image1, 1) ) )
    composite[:, :] = image1[:, :]
    composite[size:, size:] = image2[size:, size:]
    return composite

def make_scalebar( val, scale, ax ):
    '''
    Make a scalebar.

    Parameters
    ----------
    val : float
        Size of label.
    scale : float
        Scale of image.
    ax : ax
        Taget matplotlib ax.

    Returns
    -------
    scalebar

    Notes
    -----
    Creates a matplotlib scalebar that can be displayed on an image. 

    '''
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    sizelabel=str( val ) + ' nm-1'
    scalebar = AnchoredSizeBar(ax.transData,
                            (val/scale), sizelabel, 'lower left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=1)
    return scalebar

def find_iradius_itheta( image, scale ):
    '''
    Find the distance from the centre and radial angle of each pixel in an 
    array.

    Parameters
    ----------
    image : array-like
        Input image.
    scale : float
        Scale of image.

    Returns
    -------
    iradius : array-like
        Distance from center. 
    itheta : array-like
        Radial angle.

    Notes
    -----
    Returns two arrays which contain the distance from the center of the
    array (r), and the radial angle (phi) of each element in the input array.
    This allows calulation in polar coordinates of individual elements in the
    input array by referencing the corresponding elements of iradius and
    itheta using cartesian coordinates.

    As iradius is calculated with atan2, the angle varies from -pi to pi. The
    angle is minimum at 9 o'clock and increases clockwise.
    '''
    imageX = np.size( image, 1)
    imageY = np.size( image, 0)
    radius = imageX/2
    CTF2d = np.ones((imageX,imageY))
    irow, icol = np.indices( image.shape )
    centX = irow - image.shape[0] / 2.0
    centY = icol - image.shape[1] / 2.0
    #distance from centre
    iradius = ((centX**2 + centY**2)**0.5) * scale
    #angle from centre
    itheta = np.arctan2(centX, centY)
    return iradius, itheta


# line profiles
class LineProfiles:
    '''
    Class to hold line profiles.

    Attributes
    ----------
    radial_profile : array-like
        Radial profile of array.
    frequency : array-like
        Radial profile of array frequency range.
    smoothed_profile : array-like
        Smoothed radial profile.
    baseline : array-like
        Baseline of radial profile.
    cropped_profile : array-like
        Radial profile cropped to a frequency range.
    cropped_frequency : array-like
        Cropped frequency range of radial profile.
    bins : int
        Number of bins. 

    Notes
    -----
    Used by the following classes: CTF_profile, twofoldAstigmatism.

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
    Class to hold lens aberrations.

    Attributes
    ----------
    piston : float
    tilt : float
    defocus : float
    C20 : float
        Defocus alias.
    C12 : array-like
        Twofold astigmatism, as [defocus, angle]
    twofold_astigmatism : float
        C12 alias.
    Cs : float
        Spherical aberration.
    spherical_aberration : float
        Cs alias
    C30 : float
        Spherical aberration.
    C3 : float
        Cs alias

    Notes
    -----
    Names are aliased with common names. Used by following classes: CTF_image, 
    twofoldAstigmatism. 

    '''
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
    Class to hold locations of minima in CTFs.

    Attributes
    ----------
    maxima : 
    minima : 
    x_min : 
    y_min : 
    indicies_min : 
    indicies_max : 
    results : 

    Notes
    -----
    Used by following classes: CTF_image, twofoldAstigmatism. 

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