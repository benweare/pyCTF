'''
A class to contain contrast transfer functions (CTFs).
'''

import numpy as np

import matplotlib.pyplot as plt

import scipy

from pyCTF.utils import LineProfiles
from pyCTF.utils import ZerosData
from pyCTF.utils import LensAberrations

from pyCTF.utils import find_iradius_itheta
from pyCTF.utils import kv_to_lamb
from pyCTF.utils import make_scalebar

from pyCTF.profile import Profile
from pyCTF.zeros import Zeros

from pyCTF.simulation import CTFSimulation2D

from pyCTF.astig import Astig

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
    CTF = ElectronImage( image, kV, scale, LineProfiles, ZerosData )
    return CTF


class ElectronImage:
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
        self.polar = None
        self.correlation = None
        self.maximum = None
        self.minimum = None
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
        self.cropped_frequency = _process_profile(self, 
                                                    f_limits=f_limits,
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
def _process_profile( ElectronImage, **kwargs ):
    f_limits = kwargs.get( 'f_limits', [0, 5.0] )
    polynomial = kwargs.get( 'polynomial', 20 )
    window = kwargs.get( 'window', 1 )
    # kwargs to allow astigmatism defocus measurement
    rprof = kwargs.get( 'rprof', ElectronImage.radial_profile )
    freq = kwargs.get( 'freq', ElectronImage.frequency )
    baseline = kwargs.get( 'baseline', ElectronImage.baseline )
    sprof = kwargs.get( 'sprof', ElectronImage.smoothed_profile )
    cprof = kwargs.get( 'cprof', ElectronImage.cropped_profile )
    cfreq = kwargs.get( 'cfreq', ElectronImage.cropped_frequency )
    image = kwargs.get( 'image', ElectronImage.image )

    ElectronImage.polynomial=polynomial
    ElectronImage.window=window
    
    rprof, _ = Profile.radial_profile( image,
                                        ElectronImage.centX,
                                        ElectronImage.centY )
    freq, _ = Profile.radial_profile( ElectronImage.iradius,
                                    ElectronImage.centX,
                                    ElectronImage.centY )
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
def _find_zeros( ElectronImage, **kwargs):
    x_lim = kwargs.get( 'xlim', [0.0, ElectronImage.max_freq_inscribed] )
    y_lim = kwargs.get( 'ylim', [-1.0, 1.0] )
    # First index to use.
    start = kwargs.get( 'start', 2 )
    # Under or overfocus.
    underfocus = kwargs.get( 'underfocus', True )
    # kwargs so can use for astigmatism.
    minima = kwargs.get( 'minima', ElectronImage.minima )
    maxima = kwargs.get( 'maxima', ElectronImage.maxima )
    sprof = kwargs.get( 'sprof', ElectronImage.smoothed_profile )
    cfreq = kwargs.get( 'freq', ElectronImage.cropped_frequency )
    indicies_min = kwargs.get( 'indicies_min', ElectronImage.indicies_min )
    y_min = kwargs.get( 'y_min', ElectronImage.y_min )
    x_min = kwargs.get( 'x_min', ElectronImage.x_min )
    results = kwargs.get( 'results', ElectronImage.results )
    Cs = kwargs.get( 'Cs', ElectronImage.Cs )
    defocus = kwargs.get( 'defocus', ElectronImage.defocus )
    cprof = kwargs.get( 'cprof', ElectronImage.cropped_profile )
    freq = kwargs.get( 'cprof', ElectronImage.frequency )

    ElectronImage.xlim=x_lim
    ElectronImage.ylim=y_lim

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
    results, Cs, defocus = Zeros.fit( x_min, y_min, ElectronImage.lamb )
    return indicies_min, y_min, x_min, results, Cs, defocus, minima, maxima


def print_Cs_results( ElectronImage, **kwargs ):
    verbose = kwargs.get( 'verbose', True)
    Zeros.plot_figure( ElectronImage.cropped_frequency, 
                        ElectronImage.smoothed_profile, 
                        ElectronImage.minima, 
                        ElectronImage.x_min, 
                        ElectronImage.y_min, 
                        ElectronImage.results,
                        ElectronImage.cropped_frequency,
                        (ElectronImage.cropped_profile-ElectronImage.baseline),
                        ElectronImage.indicies_min)
    
    if (verbose == True):
        Zeros.print_results( ElectronImage,
                            ElectronImage.polynomial,
                            ElectronImage.window,
                            ElectronImage.xlim,
                            ElectronImage.ylim,
                            ElectronImage.defocus,
                            ElectronImage.Cs,
                            ElectronImage.results )
    return


# Wrapper to measure defocus.
def measure_defocus( ElectronImage, **kwargs ):
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
    xlim = kwargs.get( 'xlim', [0.0, ElectronImage.max_freq_inscribed] )
    ylim = kwargs.get( 'ylim', [-1.0, 1.0] )
    start = kwargs.get( 'start', 2 )
    underfocus = kwargs.get( 'underfocus', True )
    ElectronImage.radial_profile,\
    ElectronImage.frequency,\
    ElectronImage.baseline,\
    ElectronImage.smoothed_profile,\
    ElectronImage.cropped_profile,\
    ElectronImage.cropped_frequency = \
    _process_profile( ElectronImage, f_limits=f_limits, polynomial=polynomial,\
        window=window )
    ElectronImage.indicies_min,\
    ElectronImage.y_min,\
    ElectronImage.x_min,\
    ElectronImage.results,\
    ElectronImage.Cs,\
    ElectronImage.defocus,\
    ElectronImage.minima,\
    ElectronImage.maxima = \
    _find_zeros( ElectronImage, xlim=xlim,ylim=ylim,start=start,\
        underfocus=underfocus )
    return


#WIP
def _phase_plate( ElectronImage ):
    # Aperture.
    plate = CTFSimulation2D( ElectronImage.max_freq_inscribed*2,
                            int(ElectronImage.length),
                            ElectronImage.kV, -500)
    plate.defocus = ElectronImage.defocus
    plate.C12a = ElectronImage.C12a
    plate.C12b = ElectronImage.C12b
    plate.phi = ElectronImage.phi
    plate.Cs = ElectronImage.Cs
    plate.update()
    fig, ax = plt.subplots(1)
    ax.matshow( plate.square_CTF )
    #Make and show a phase plate.
    #Under development. Function to take aberrations in CTF object and
    #return a phase plate.
    return