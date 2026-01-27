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

from pyCTF.ctf_profile import Profile
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
    CTF = CTFImage( image, kV, scale, LineProfiles, ZerosData )
    return CTF

class CTFImage:
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
    show_image( self )
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
        #return

    def astig_magnitude( self, defocus_guess, **kwargs ):
        '''
        Wrapper to measure twofold astigmatism magnitude.

        Parameters
        ----------
        defocus_guess : float
            Estimate of defocus in nm. 
        phi : float, optional
            Known angle of astigmatism.
        astig_max : float, optional
            Maximum value of twofold astigmatism to apply.
        slices : int, optional
            Number of slices, defaults to 11.

        Notes
        -----
        Wrapper around methods in twofoldAstigmatism class to measure the
        astigmatism in the CTF.
        '''
        phi = kwargs.get( 'phi', self.astig.amax )
        astig_max = kwargs.get( 'max_val', 1000 )
        slices = kwargs.get( 'slices', 11 )
        CTF = self
        
        # Multiply max freq by 2 as we need the diameter not the radius.
        CTF2D = CTFSimulation2D( CTF.max_freq_inscribed*2,
                                int(CTF.length),
                                CTF.kV, -500)
        CTF2D.scale = CTF.scale
        CTF2D.defocus = defocus_guess * 1e-9
        CTF2D.phi = np.deg2rad( phi )
        CTF2D.update()
    
        vals, a, polar_list = self.astig.magnitude_measure( CTF.image, 
                                                            slices, 
                                                            astig_max,
                                                            CTF2D,
                                                            radius=int(np.round((CTF.length/2))) )
        x, y = self.astig.find_astig_defocus( vals, a )
        
        CTF2D.C12a = x[0] *1e-9
        CTF2D.C12b = y[0] *1e-9
        CTF2D.update()
        
        self.astig.plot_results( vals, slices, a, CTF2D )
        print( str(x[0]) + ', ' + str(y[0]) )
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
    def __process_profile( self, **kwargs ):
        f_limits = kwargs.get( 'f_limits', [0, 5.0] )
        polynomial = kwargs.get( 'polynomial', 20 )
        window = kwargs.get( 'window', 1 )
        # kwargs to allow astigmatism defocus measurement
        rprof = kwargs.get( 'rprof', self.radial_profile )
        freq = kwargs.get( 'freq', self.frequency )
        baseline = kwargs.get( 'baseline', self.baseline )
        sprof = kwargs.get( 'sprof', self.smoothed_profile )
        cprof = kwargs.get( 'cprof', self.cropped_profile )
        cfreq = kwargs.get( 'cfreq', self.cropped_frequency )
        image = kwargs.get( 'image', self.image )

        self.polynomial=polynomial
        self.window=window
        
        rprof, _ = Profile.radial_profile( image,
                                            self.centX,
                                            self.centY )
        freq, _ = Profile.radial_profile( self.iradius,
                                        self.centX,
                                        self.centY )
        try:
            cfreq, cprof = Profile.crop_frequency( rprof, freq, f_limits )
        except:
            cfreq = freq
            cprof = rprof
            print( 'Error: could not crop to frequency range.\n' )
        baseline = Profile.remove_baseline( cprof )
        sprof = Profile.smooth_profile( ( cprof - baseline ), polynomial, window )
        return rprof, freq, baseline, sprof, cprof, cfreq

    def plot_radial_profiles( self ):
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


    def process_CTF_profile( self, **kwargs ):
        '''
        Wrapper around process profile that is more convient to use.
        '''
        f_limits = kwargs.get( 'f_limits', [0, 5.0] )
        polynomial = kwargs.get( 'polynomial', 20 )
        window = kwargs.get( 'window', 1 )
        self.radial_profile,\
        self.frequency,\
        self.baseline,\
        self.smoothed_profile,\
        self.cropped_profile,\
        self.cropped_frequency = self.__process_profile(f_limits=f_limits,
                                                    polynomial=polynomial,
                                                    window=window)
        return
        

    # Find the minima in the CTF radial profile.
    def __find_zeros( self, **kwargs):
        x_lim = kwargs.get( 'xlim', [0.0, self.max_freq_inscribed] )
        y_lim = kwargs.get( 'ylim', [-1.0, 1.0] )
        # First index to use.
        start = kwargs.get( 'start', 2 )
        # Under or overfocus.
        underfocus = kwargs.get( 'underfocus', True )
        # kwargs so can use for astigmatism.
        minima = kwargs.get( 'minima', self.minima )
        maxima = kwargs.get( 'maxima', self.maxima )
        sprof = kwargs.get( 'sprof', self.smoothed_profile )
        cfreq = kwargs.get( 'freq', self.cropped_frequency )
        indicies_min = kwargs.get( 'indicies_min', self.indicies_min )
        y_min = kwargs.get( 'y_min', self.y_min )
        x_min = kwargs.get( 'x_min', self.x_min )
        results = kwargs.get( 'results', self.results )
        Cs = kwargs.get( 'Cs', self.Cs )
        defocus = kwargs.get( 'defocus', self.defocus )
        cprof = kwargs.get( 'cprof', self.cropped_profile )
        freq = kwargs.get( 'cprof', self.frequency )

        self.xlim=x_lim
        self.ylim=y_lim

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
        results, Cs, defocus = Zeros.fit( x_min, y_min, self.lamb )
        return indicies_min, y_min, x_min, results, Cs, defocus, minima, maxima

    def print_Cs_results( self, **kwargs ):
        verbose = kwargs.get( 'verbose', True)
        Zeros.plot_figure( self.cropped_frequency, 
                            self.smoothed_profile, 
                            self.minima, 
                            self.x_min, 
                            self.y_min, 
                            self.results,
                            self.cropped_frequency,
                            (self.cropped_profile-self.baseline),
                            self.indicies_min)
        
        if (verbose == True):
            Zeros.print_results( self,
                                self.polynomial,
                                self.window,
                                self.xlim,
                                self.ylim,
                                self.defocus,
                                self.Cs,
                                self.results )
        return

    def __phase_plate( CTF ):
        #Make and show a phase plate.
        #Under development. Function to take aberrations in CTF object and
        #return a phase plate.
        return

    # convienience function to measure defocus
    def measure_defocus( self, **kwargs ):
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
        self.radial_profile,\
        self.frequency,\
        self.baseline,\
        self.smoothed_profile,\
        self.cropped_profile,\
        self.cropped_frequency = \
        self.__process_profile( f_limits=f_limits, polynomial=polynomial,\
            window=window )
        self.indicies_min,\
        self.y_min,\
        self.x_min,\
        self.results,\
        self.Cs,\
        self.defocus,\
        self.minima,\
        self.maxima = \
        self.__find_zeros( xlim=xlim,ylim=ylim,start=start,\
            underfocus=underfocus )
        return

    # Methods for astigmatism measurement.
    def astig_angle( self ):
        '''
        Measure angle of twofold astigmatism in CTF.

        Notes
        -----
        Wrapper to streamline measuring astigmatism to a single call.
        '''
        # check for profiles
        self.radial_profile, _ = Profile.radial_profile( self.image, self.centX, self.centY )
        self.frequency, _ = Profile.radial_profile( self.iradius, self.centX, self.centY )
        self.astig.measure_angle()
        return

    def print_astig_results( self ):
        '''
        Show outcome of twofold astigmatism measurement.

        Notes
        -----
        Wrapper around Zeros.plotCsFigure() and Zeros.printResults().
        '''
        n = self.astig
        Zeros.plotCsFigure( n.cropped_frequency, 
                            n.smoothed_profile, 
                            n.minima, 
                            n.x_min, 
                            n.y_min, 
                            n.results,
                            n.cropped_frequency,
                            (n.cropped_profile-n.baseline) )
        
        Zeros.printResults( n.defocus, 
                                n.Cs, 
                                n.results )
        return

    ## Method to measure astigmatism via sectors method, not in use.
    def __astig_defocus( self, angle, **kwargs ):
        polynomial = kwargs.get( 'polynomial', 20 )
        window = kwargs.get( 'window', 1 )
        f_limits = kwargs.get( 'flim', [0.0, 3.0] )
        xlim = kwargs.get( 'xlim', [0.0, None] )
        ylim = kwargs.get( 'ylim', [0.0, None] )
        start = kwargs.get( 'start', 2 )
        underfocus = kwargs.get( 'underfocus', True )
        width = kwargs.get( 'width', 25 )
    
        n = self.astig
        # mask
        n.mask( angle, width )
        # defocus
        n.radial_profile, n.frequency, n.baseline, n.smoothed_profile, n.cropped_profile, n.cropped_frequency = \
        self.__process_profile( f_limits=f_limits, polynomial=polynomial, window=window, image=n.masked,
                              rprof=n.radial_profile, freq=n.frequency, sprof=n.smoothed_profile, cprof=self.cropped_profile,
                              baseline=n.baseline)

        n.smoothed_profile = Profile.smooth_profile( n.smoothed_profile, 20, 1 )

        n.indicies_min, n.y_min, n.x_min, n.results, n.Cs, n.defocus, n.minima, n.maxima = \
        self.__find_zeros( xlim=xlim, 
                         ylim=ylim, 
                         start=start, 
                         underfocus=underfocus, 
                         sprof=n.smoothed_profile, freq=n.frequency, cfreq=n.cropped_frequency,
                         y_min=n.y_min, xmin=n.y_min, indicies_min=n.indicies_min, 
                         minima=n.minima, maxima=n.maxima)
        return