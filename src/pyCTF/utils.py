'''
Module containing miscellanous functions. 

This module contains miscellanous functions used by other  modules 
in the PyCTF package.
'''

#from numba import jit, config

#from pyCTF.__init__ import enable_jit
#if enable_jit == True:
#    config.DISABLE_JIT = True
#else:
#    config.DISABLE_JIT = False

import numpy as np
from numpy.polynomial import polynomial

import scipy
from scipy.constants import( e, c, m_e, h )

#@jit
def scherzer_defocus( input ):
    '''
    Scherzer defocus, in nanometers.
    '''
    scherzer = (-4/3) * np.sqrt( input.Cs * input.lamb )
    return scherzer*1e9

# Lichte defocus.
#def lichte_defocus():
#    lichte = (-3/4) * Cs * (R * lamb**2)
#    return lichte

#@jit
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

    """
    E = kV*1000
    PT = scipy.constants.h * scipy.constants.c
    PBA = (scipy.constants.e *E)*(scipy.constants.e *E)
    PBB = 2*scipy.constants.e*E*scipy.constants.m_e*(scipy.constants.c)\
    *(scipy.constants.c)
    # Wavelength in meters.
    lamb = PT/np.sqrt(PBA+PBB)
    return lamb

#@jit
def normalise_data_range( data, dmin=0, dmax=1 ):
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
    #dmin = kwargs.get('dmin', 0)
    #dmax = kwargs.get('dmax', 1)
    return ((data-np.min(data))/(np.max(data)-np.min(data)))*( dmax - dmin )

def baseline_als( y, lam, p, **kwargs ):
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
    from scipy.sparse.linalg import spsolve 
    from scipy import sparse
    n_iter = kwargs.get('n_iter', 10)
    L = len( y )
    D = sparse.diags([1,-2,1],
                    [0,-1,-2],
                    shape=(L,L-2),
                    dtype='float',
                    format='csr')
    w = np.ones( L )
    for i in range( n_iter ):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


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
        Size of image, i.e. CTF.width
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


# To do: try to get scale from CTF class by default?
def show_image( image, **kwargs ):
        '''
        Display an image of the CTF.

        Parameters
        ----------
        scalebar : bool, optional
            Add a scalebar to image if True.
        length : float, optional
            Length of scalebar in nm-1.

        Notes
        -----
        Uses plt.matshow to display the image.
        '''
        import matplotlib.pyplot as plt
        scale = kwargs.get( 'scale', 0 )
        val = kwargs.get( 'length', 0.5 )
        cbar = kwargs.get('cbar', True)
        norm = kwargs.get('norm', True)
        fig, ax = plt.subplots()
        if norm == True:
            cax = ax.matshow( normalise_data_range(image) )
        else:
            cax=ax.matshow( image )
        ax.set_xticks([])
        ax.set_yticks([])
        if ( scale != 0 ):
            try:
                from pyCTF.utils import make_scalebar
                scalebar = make_scalebar( val, scale, ax )
                ax.add_artist(scalebar)
            except:
                print('Error: could not add scalebar to image.')
        if cbar == True:
            try:
                cbar = fig.colorbar( mappable=cax )
            except:
                print('Error: could not add colourbar to image.')
        plt.show()
        return


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


#@jit#(debug=True)
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
    imageX = image.shape[1] #np.size( image, np.int64(1) )
    imageY = image.shape[0]#np.size( image, 0)
    radius = imageX/2
    CTF2d = np.ones((imageX,imageY))
    irow, icol = np.indices( image.shape )
    centX = irow - image.shape[0] / 2.0
    centY = icol - image.shape[1] / 2.0
    # Distance from centre.
    iradius = ((centX**2 + centY**2)**0.5) * scale
    # Angle from centre.
    itheta = np.arctan2(centX, centY)
    return iradius, itheta


# function to bin images.
def bin_array(data, binstep=2, binsize=2, func=np.sum):
    '''
    Function to bin arrays by an arbitary integer.

    Default is 2x2 binning.

    Parameters
    ----------
    data : array
    binstep : int
    binsize : int
    func : numpy method
    '''
    # Function to bin array with numpy. Default is 2x binning.
    # See: https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    axes = [0, 1]
    data = np.array(data)

    remainder =  np.mod( len(data[0]), binstep )

    if remainder != 0:
        print('\nCropping array to size ' + str(len(data[0])-remainder) )
        try:
            data = crop_array( data, (len(data[0])-remainder) )
        except:
            print('\nError: could not crop array.')

    dims = np.array(data.shape)

    for axis in axes:
        argdims = np.arange(data.ndim)
        argdims[0], argdims[axis]= argdims[axis], argdims[0]
        data = data.transpose(argdims)
        data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]/binstep)]
        data = np.array(data).transpose(argdims)
    return data


def crop_array( image, width, **kwargs ):
        '''
        Centre-crop and array to a specified size.

        Parameters
        ----------
        image : array
        width : int
        zstart : int, optional
        zend : int, optional

        Returns
        -------
        out : array

        Notes
        -----
        Used to crop Fourier transform to centre region containing contrast
        transfer function.
        '''
        zstart = kwargs.get( 'zstart', 0 )
        zend = kwargs.get( 'zend', None )
        centX = len(image[0])/2
        centY = len(image[1])/2
        # don't let the axis be padded
        if ( width > len(image[0]) ):
            width = len(image[0])
        # slice image
        xstart = round( centX - (width/2) )
        xend = round( centX + (width/2) )
        ystart = round( centY - (width/2) )
        yend = round( centY + (width/2) )
        if ( image.ndim == 3 ):
            out = image[ xstart:xend, ystart:yend, zstart:zend ]
        else:
            out = image[ xstart:xend, ystart:yend ]
        return out


def fit( x_min, y_min, lamb ):
    '''
    Fit gradient for spherical aberration using Numpy.

    Parameters
    ----------
    x_min : array
    y_min : array
    lamb : float

    Returns
    -------
    intercept : float
    slope : float
    Cs : float
    defocus : float

    Notes
    -----
    Redundant with zeros.fit_numpy(), but does not calculate Cs or defocus.
    '''
    [intercept, slope] = polynomial.polyfit(x_min, y_min, 1, full=False )
    # covariance
    cov = np.sqrt( np.diagonal( np.cov( x_min, y_min )))
    return slope, intercept, cov


# Calculate the Cs and defocus from the gradient and y-intercept.
# TO DO: move to utils.
def calc_cs_and_defocus( m, c, lamb ):
    Cs = m / ( lamb**3 )
    defocus = -c /( -2 * lamb )
    return Cs, defocus


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

    def _float_to_rgb( image ):
        return
