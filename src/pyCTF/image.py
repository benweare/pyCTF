'''
A class to contain contrast transfer functions (CTFs).
'''

import numpy as np

import matplotlib.pyplot as plt

import scipy

from pyCTF.misc import LineProfiles
from pyCTF.misc import ZerosData
from pyCTF.misc import LensAberrations

from pyCTF.misc import find_iradius_itheta
from pyCTF.misc import kv_to_lamb
from pyCTF.misc import make_scalebar

from pyCTF.profile import Profile
from pyCTF.zeros import Zeros

from pyCTF.simulation import CTFSimulation2D

from pyCTF.twofold_astigmatism import twofoldAstigmatism

#import misc
#import twofold_astigmatism
#import zeros

class filterError( Exception ):
    '''
    Exception raised when Zeros.filter_zeros() fails, or filters all data out.

    Attributes
    ----------
    message : string

    Notes
    -----
    If minima if returned with 0 length, try changing the limits applied
    during filtering. 

    '''
    def __init__( self, message ):
        self.message = message
        return


def import_ctf( image, kV, scale ):
    '''
    Import experimental CTF as CTF_image class.

    Parameters
    ----------
    image : array
    kV : float
        Accelerating voltage in kilovolts.
    scale : float 
        Image scale in nm per pixel.

    Returns
    -------
    CTF : class

    Notes
    -----
    Wrapper around CTF_image class declaration to make it easier to create
    new CTF objects.
    '''
    CTF = Image( image, kV, scale, LineProfiles, ZerosData )
    return CTF


class Image:
    '''
    Class for holding and manipulating experimental CTFs.

    Attributes
    ----------
    kV : float
    lamb : float
    scale : float
    image : array
    stage_tilt : float
    LF_bkg : array
        Low-frequency background.
    E_bkg : array
        Envelope background.
    iradius : array
    itheta : array
    width : float
    length : float
    centX : float
    centY : float
    max_freq_inscribed : float
        Largest inscribed circle in image.
    max_freq : float
    astig : class
        twofoldAstigmatism class.

    Warnings
    --------
    The remove_background() method replaces the imported image with the
    background subracted image.

    Methods
    -------
    astig_magnitude( self, defocus_guess, **kwargs )
    remove_background( self, rstart1, rstart2 )
    plot_background( self )
    process_profile( self, **kwargs )
    find_zeros( self, **kwargs )
    print_Cs_results( self )
    measure_defocus( self, **kwargs )
    astig_angle( self )
    astig_defocus( self, angle, **kwargs )

    Notes
    -----
    This class is the core of pyCTF. It contains the CTF as an image, and
    allows measurement of lens aberrations by class methods. 

    Many methods of class are wrappers intended to streamline CTF processing
    by hiding the nuts and bolts (somewhat). 
    '''
    def __init__(self, image, kV, scale, LineProfiles, ZerosData ):
        '''
        Parameters
        ----------
        image : array
        kV : float
        scale : float
        line_profiles : class
        zeros_data : class

        Notes
        -----
        It is preferred to create class via import_CTF() function.
        '''
        self.kV = kV
        self.lamb = kv_to_lamb( kV )
        self.scale = scale
        self.image = image
        self.stage_tilt=None
        # Low frequency and enevelope background functions.
        self.LF_bkg = None
        self.E_bkg = None
        # itheta is -180 to 180 as it uses atan2.
        self.iradius, self.itheta = find_iradius_itheta( self.image, self.scale )
        self.width = len(image[0])
        self.length = len(image[0])
        self.centX = len(image[0])/2
        self.centY = len(image[0])/2
        # Maximum frequency of inscribed circle.
        self.max_freq_inscribed = self.scale * (self.width/2)
        # Maximum frequency (corner of image).
        self.max_freq = self.scale * np.sqrt( (self.width/2)**2\
            + (self.width/2)**2 )
        # Component classes.
        self.astig = twofoldAstigmatism( self )
        # Data structures.
        LineProfiles.__init__( self )
        LensAberrations.__init__( self )
        ZerosData.__init__( self )
        self.polynomial=None
        self.window=None
        self.xlim=None
        self.ylim=None
        # Astigmatism values
        self.radius = None
        self.masked = None
        self.polar = None
        # defocus of astigmatism
        self.fmax = None
        self.fmin = None
        # angle of astigmatism
        self.amin = None
        self.amax = None
        return

    def remove_background( self, rstart1, rstart2 ):
    '''
    Remove CTF background using Fourier methods.

    Parameters
    ----------
    rstart1 : float
    rstart2 : float

    Notes
    -----
    Wrapper around Fourier.remove_bckg() that passes class attributes to
    method.

    See Also
    --------
    Fourier.remove_bckg()
    CTFImage.plot_background()

    '''

    from pyCTF.fourier import Fourier

    self.image, self.LF_bkg, self.E_bkg = Fourier.remove_bckg( self.image, 
                                                                rstart1, 
                                                                rstart2 )
    return


    def get_profiles( self, **kwargs ):
    '''
    Wrapper around __process_profile that is more convient to use.
    '''
    f_limits = kwargs.get( 'f_limits', [0, 5.0] )
    polynomial = kwargs.get( 'polynomial', 20 )
    window = kwargs.get( 'window', 1 )
    self.radial_profile,\
    self.frequency,\
    self.baseline,\
    self.smoothed_profile,\
    self.cropped_profile,\
    self.cropped_frequency = __process_profile(f_limits=f_limits,
                                                polynomial=polynomial,
                                                window=window)
    return


    def plot_profiles( self ):
    '''
    Show the results of measuring radial profiles.
    '''
    fig, axs = plt.subplots( 1, 2, figsize=(8,8) )
    axs[0].plot( self.frequency, self.radial_profile, label='Radial profile' )
    axs[0].plot( self.cropped_frequency, self.baseline, label='Baseline' )
    axs[0].plot( self.cropped_frequency, self.cropped_profile, label='Cropped' )
    axs[1].plot( self.cropped_frequency, self.smoothed_profile, label='Smoothed profile')
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    axs[0].set_ylabel('Intensity / a.u.', fontsize = 16)
    axs[0].set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
    axs[1].set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
    axs[0].legend()
    axs[1].legend()
    return


    def plot_background( self ):
        '''
        Plot results of Fourier background removal.

        Notes
        -----
        Creates a plot showing the processed CTF, the low frequency
        background, and the envelope background.
        '''
        fig, axs = plt.subplots(1, 3, figsize=(8,8))
        axs[0].matshow( self.image )
        axs[1].matshow( self.LF_bkg )
        axs[2].matshow( self.E_bkg )

        axs[0].set_title( 'Background subtracted' )
        axs[1].set_title( 'Low frequency' )
        axs[2].set_title( 'Envelope' )

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        try:
            scalebar = make_scalebar( 0.5, self.scale, axs[0] )
            axs[0].add_artist(scalebar)
            scalebar = make_scalebar( 0.5, self.scale, axs[1] )
            axs[1].add_artist(scalebar)
            scalebar = make_scalebar( 0.5, self.scale, axs[2] )
            axs[2].add_artist(scalebar)
        except:
            print('Error: could not add scalebar to image.')
        return


# Extract and process the radial profile of the CTF.
def __process_profile( Image, **kwargs ):
    f_limits = kwargs.get( 'f_limits', [0, 5.0] )
    polynomial = kwargs.get( 'polynomial', 20 )
    window = kwargs.get( 'window', 1 )
    # kwargs to allow astigmatism defocus measurement
    rprof = kwargs.get( 'rprof', Image.radial_profile )
    freq = kwargs.get( 'freq', Image.frequency )
    baseline = kwargs.get( 'baseline', Image.baseline )
    sprof = kwargs.get( 'sprof', Image.smoothed_profile )
    cprof = kwargs.get( 'cprof', Image.cropped_profile )
    cfreq = kwargs.get( 'cfreq', Image.cropped_frequency )
    image = kwargs.get( 'image', Image.image )

    Image.polynomial=polynomial
    Image.window=window
    
    rprof, _ = Profile.radial_profile( image,
                                        Image.centX,
                                        Image.centY )
    freq, _ = Profile.radial_profile( Image.iradius,
                                    Image.centX,
                                    Image.centY )
    try:
        cfreq, cprof = Profile.crop_frequency( rprof, freq, f_limits )
    except:
        cfreq = freq
        cprof = rprof
        print( 'Error: could not crop to frequency range.\n' )
    baseline = Profile.remove_baseline( cprof )
    sprof = Profile.smooth_profile( ( cprof - baseline ), polynomial, window )
    return rprof, freq, baseline, sprof, cprof, cfreq


# Find the minima in the CTF radial profile.
def __find_zeros( Image, **kwargs):
    x_lim = kwargs.get( 'xlim', [0.0, Image.max_freq_inscribed] )
    y_lim = kwargs.get( 'ylim', [-1.0, 1.0] )
    # First index to use.
    start = kwargs.get( 'start', 2 )
    # Under or overfocus.
    underfocus = kwargs.get( 'underfocus', True )
    # kwargs so can use for astigmatism.
    minima = kwargs.get( 'minima', Image.minima )
    maxima = kwargs.get( 'maxima', Image.maxima )
    sprof = kwargs.get( 'sprof', Image.smoothed_profile )
    cfreq = kwargs.get( 'freq', Image.cropped_frequency )
    indicies_min = kwargs.get( 'indicies_min', Image.indicies_min )
    y_min = kwargs.get( 'y_min', Image.y_min )
    x_min = kwargs.get( 'x_min', Image.x_min )
    results = kwargs.get( 'results', Image.results )
    Cs = kwargs.get( 'Cs', Image.Cs )
    defocus = kwargs.get( 'defocus', Image.defocus )
    cprof = kwargs.get( 'cprof', Image.cropped_profile )
    freq = kwargs.get( 'cprof', Image.frequency )

    Image.xlim=x_lim
    Image.ylim=y_lim

    minima, maxima = Zeros.calc_zeros( sprof )

    try:
        minima = Zeros.filter_zeros( minima, 
                                     sprof, 
                                     cfreq, 
                                     x_lim, 
                                     y_lim )
    except:
        message = 'Error: CTF minima not filtered.'
        #raise filterError( message )
        print( message )
    # Exception rasied if all minima are filtered.
    if (len( minima ) == 0 ):
        message = 'Error: all minima filtered. Try checking radial profile?'
        raise filterError( message )
        return
    indicies_min, x_min, y_min = Zeros.calc_indicies( minima, sprof,\
        cfreq, start=start, underfocus=underfocus)
    results, Cs, defocus = Zeros.fit( x_min, y_min, Image.lamb )
    return indicies_min, y_min, x_min, results, Cs, defocus, minima, maxima


def print_Cs_results( Image, **kwargs ):
    verbose = kwargs.get( 'verbose', True)
    Zeros.plot_figure( Image.cropped_frequency, 
                        Image.smoothed_profile, 
                        Image.minima, 
                        Image.x_min, 
                        Image.y_min, 
                        Image.results,
                        Image.cropped_frequency,
                        (Image.cropped_profile-Image.baseline),
                        Image.indicies_min)
    
    if (verbose == True):
        Zeros.print_results( Image,
                            Image.polynomial,
                            Image.window,
                            Image.xlim,
                            Image.ylim,
                            Image.defocus,
                            Image.Cs,
                            Image.results )
    return


# Wrapper to measure defocus.
def measure_defocus( Image, **kwargs ):
    '''
    Measure the defocus of a CTF.

    Parameters
    ----------
    polynomial : int, optional
        Savitsky-Golay polynomial.
    window : int, optional
        Savitsky-Golay window.
    f_limits : array, optional
        Array as [f1, f2].
    xlim : array, optional
        Array as [x1, x2].
    ylim : array, optional
        Array as [y1, y2].
    start : int, optional
        First integer to use when labelling minima.
    underfocus : bool, optional
        True for underfocus, False for overfocus.

    Warnings
    --------
    Non-linear relationship may arise from incorrect index assignment by
    Zeros.calc_indicies(), controlled by 'start' argument.

    Notes
    -----
    Wrapper to streamline the process of measuring defocus in a CTF, using 
    standard values. 
    '''
    polynomial = kwargs.get( 'polynomial', 20 )
    window = kwargs.get( 'window', 1 )
    f_limits = kwargs.get( 'f_limits', [0.0, 5.0] )
    xlim = kwargs.get( 'xlim', [0.0, self.max_freq_inscribed] )
    ylim = kwargs.get( 'ylim', [-1.0, 1.0] )
    start = kwargs.get( 'start', 2 )
    underfocus = kwargs.get( 'underfocus', True )
    Image.radial_profile,\
    Image.frequency,\
    Image.baseline,\
    Image.smoothed_profile,\
    Image.cropped_profile,\
    Image.cropped_frequency = \
    Image.__process_profile( f_limits=f_limits, polynomial=polynomial,\
        window=window )
    Image.indicies_min,\
    Image.y_min,\
    Image.x_min,\
    Image.results,\
    Image.Cs,\
    Image.defocus,\
    Image.minima,\
    Image.maxima = \
    Image.__find_zeros( xlim=xlim,ylim=ylim,start=start,\
        underfocus=underfocus )
    return


#WIP
def __phase_plate( Image ):
    # Aperture.
    plate = CTFSimulation2D( Image.max_freq_inscribed*2,
                            int(Image.length),
                            Image.kV, -500)
    plate.defocus = Image.defocus
    plate.C12a = Image.C12a
    plate.C12b = Image.C12b
    plate.phi = Image.phi
    plate.Cs = Image.Cs
    plate.update()
    fig, ax = plt.subplots(1)
    ax.matshow( plate.square_CTF )
    #Make and show a phase plate.
    #Under development. Function to take aberrations in CTF object and
    #return a phase plate.
    return