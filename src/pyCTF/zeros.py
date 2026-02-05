'''
A class to measure CTF zeros with contained methods.
'''

import numpy as np 
import scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import pyCTF.misc
from pyCTF.misc import gradient_simple

class indicieError( Exception ):
    '''
    Exceptions raised for custom error scenarios.

    Attributes
    ----------
    message : string
    '''
    def __init__( self, message ):
        self.message = message
        return

# class to hold methods for determining defocus and Cs via zeros method
class Zeros:
    '''
    Methods for measuring minima of CTFs.

    Methods
    -------
    fit_gradient( x_min, y_min, lamb )
    calc_zeros( data )
    filter_zeros( minima, prof, freq, xlim, ylim )
    calc_indicies( minima, data, freq, **kwargs )
    indicies( length, **kwargs )
    plot_zeros( minima, data, freq )
    fit( x_min, y_min, lamb )
    plot_figure( x, y, minima, x_min, y_min, results, xraw, yraw )
    print_results( defocus, Cs, results )

    Notes
    -----
    Uses literature method to determine defocus and spherical aberration from
    frequency of CTF minima. Used in conjunction with radial_profiles class.
    '''
    def __init__( self ):
        return

    def fit_gradient( x_min, y_min, lamb ):
        '''
        Fit gradient for spherical aberration.

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
        '''
        from numpy.polynomial import polynomial as P
        [intercept, slope] = P.polyfit(x_min, y_min, 1, full=False )
        # covariance
        cov = np.sqrt( np.diagonal( np.cov( x_min, y_min )))
        ## Cs and defocus
        Cs = slope / ( lamb**3 )
        defocus = -intercept /( -2 * lamb )
        return intercept, slope, Cs, defocus
    
    def calc_zeros( data ):
        '''
        Find maxima and minima of array.

        Parameters
        ----------
        data : array

        Returns
        -------
        minima : array
        maxima : array

        Notes
        -----
        Uses scipy.signal.find_peaks to find maxima and minima of data. 
        '''
        minima, _ = scipy.signal.find_peaks( -data )
        maxima, _ = scipy.signal.find_peaks( data )
        return minima, maxima

    def filter_zeros( minima, prof, freq, xlim, ylim ):
        '''
        Filter data by passed limts. 

        Parameters
        ----------
        minima : array
        prof : array
        freq : array
        xlim : array
            Array as [x0, x1].
        ylim : array
            Array as [y0, y2].

        Returns
        -------
        minima : array
            Filtered data.

        Notes
        -----
        Filters input data to discard any datapoints outside of the provided 
        x- and y-axis range. 
        '''
        # freq
        #minima = self.filter_range( minima, freq, xlim[0], xlim[1] )
        minima = np.array([x for x in minima if freq[x] <= xlim[1]])
        minima = np.array([x for x in minima if freq[x] >= xlim[0]])
        # prof
        #minima = self.filter_range( minima, prof, ylim[0], ylim[1] )
        minima = np.array([x for x in minima if prof[x] <= ylim[1]])
        minima = np.array([x for x in minima if prof[x] >= ylim[0]])
        return minima

    # add indicies for maxima
    # add pos/neg for under/overfocus
    def calc_indicies( minima, data, freq, **kwargs ):
        '''
        Calculate indicies for CTF minima.

        Parameters
        ----------
        minima : array
        data : array
        freq : array
        start : int, optional
            Starting integer, if not 2.
        underfocus : bool, optional
            True for underfocus, False for overfocus.

        Returns
        -------
        indicies_min : array
        x_min : array
        y_min : array

        Notes
        -----
        Generates indicies of CTF minima for use with fitting to
        find spherical aberration and defocus, according to the
        literature conventions.
        '''
        start = kwargs.get( 'start', 2 )
        underfocus = kwargs.get( 'underfocus', True )
        indicies_min = int(-1) * Zeros.indicies( len(minima), first = start )
        if(underfocus==False):
            indicies_min = -indicies_min
        #indicies_min = np.array( list( range( 2, len( minima )*2+2, 2 ) ) )
        # indicies_max = ...
        x_min = ( freq[ minima ] )**2
        y_min = ( indicies_min ) / ( freq[ minima ]**2 )
        return indicies_min, x_min, y_min

    def indicies( length, **kwargs ):
        '''
        Indicies for CTF minima.

        Parameters
        ----------
        length : int
            Number of indicies to return.
        first : int, optional
            First indicie to return. 

        Returns
        -------
        mini : array
            Indicies of CTF minima.

        Raises
        ------
        indiciesError
            If more CTF minima than 20.

        Notes
        -----
        Generates indicies of CTF minima for use with fitting to 
        find spherical aberration and defocus, according to the 
        literature conventions. 
        '''
        first = kwargs.get( 'first', 2 )
        # supports up to 20 CTF zeros
        mini = np.array( [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40] )
        if ( length > len( mini ) ):
            raise indicieError( 'Ran out of indicies, greater than 20 elements in minima.' )
            #print( 'error: ran out of indicies' )
        #maxi = np.array( [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] )
        if ( first % 2 != 0 ):
            start = 0
            print( 'inputError: check first is even.' )
        if ( first < 0 ):
            print( 'inputError: check first is positive.' )
            start = 0
        if ( first != 2 ):
            start = int( (first / 2) - 1 )
        if ( first == 2 ):
            start = 0
        end =  start + length
        mini = mini[ start : end ]
        return mini

    def plot_zeros( minima, data, freq ):
        '''
        Plot CTF with zeros marked.

        Parameters
        ----------
        minima : array
        data : array
            y-axis data.
        freq : array
            x-axis data.
        '''
        fig, ax = plt.subplots( )
        #ax.plot( freq, cropped_profiles, alpha=0.6, linestyle='--')
        ax.plot(freq, data)
        ax.plot(freq[ minima ], data[ minima ], 'x')
        return
    
    # make sure units on results all work fine
    def fit( x_min, y_min, lamb ):
        '''
        Fit to find spherical aberration and defocus using lmfit.

        Parameters
        ----------
        x_min : array
        y_min : array
        lamb : float

        Returns
        -------
        results : class
            lmffit ModelResults class.
        Cs : float
        defocus : float

        Notes
        -----
        Uses lmfit to find the defocus and spherical aberration from an 
        experimental CTF.

        n/k**2 = k^2*0.5(Cs*lamb^3) + f*lamb
        Cs = 2m/lamb^3
        f = c/lamb
        '''
        from lmfit import Model as mdl
        model = mdl( gradient_simple )
        params = model.make_params( )
        params['m'].value = -1.0
        params['m'].vary = True
        params['m'].min = -1.0
        params['m'].max = 1.0
        params['c'].value = 0.0
        params['c'].vary = True
        results = model.fit( y_min, params, x = x_min )
        Cs = ( 2* results.params['m'].value /( ( lamb )**3)) * 1e-33 # check units conversion
        defocus = (results.params['c'].value/( lamb )) *1e-9 # check units conversion
        return results, Cs, defocus
    
    # clean up xraw and yraw
    def plot_figure( x, y, minima, x_min, y_min, results, xraw, yraw, indi_min ):
        '''
        Plot figure for spherical aberration fiting.

        Parameters
        ----------
        x : array
        y : array
        minima : array
        x_min : array
        y_min : array
        results : class
        xraw : array
        yraw : array
        '''
        fig, axs  = plt.subplots(1,2, figsize=(8,8), layout='constrained')
        axs[0].plot( xraw, yraw, alpha=0.4, label='baseline corrected' )
        axs[0].plot( x, y, label='smoothed' )
        axs[0].plot( x[ minima ], y[ minima ], "x" )
        #axs[0].set_ylim([None, 1.0])
        for i, txt in enumerate(indi_min):
            axs[0].annotate(txt, (x[minima[i]], y[minima[i]]))

        axs[1].plot( x_min, y_min, 'x', label='minima' )
        axs[1].plot( x_min, results.best_fit, label='best fit' )

        axs[0].set_box_aspect(1)
        #axs[0].set_title("Fit")
        axs[0].set_ylabel('Intensity', fontsize = 16)
        axs[0].set_xlabel('Frequency / nm-1', fontsize = 16)
        
        axs[1].set_box_aspect(1)
        axs[1].set_ylabel('$n/k_{n}^2 / nm^2$', fontsize = 16)
        axs[1].set_xlabel('$k_{n}^2 / nm^{-2}$', fontsize = 16)

        axs[0].legend()
        axs[1].legend()
        return

    def print_results(CTF, poly, win, xlim, ylim, defocus, Cs, results):
        '''
        Print results of fitting to console.
        '''
        print('-------------')
        print('lmfit results')
        print('-------------')
        print( results.fit_report(show_correl=False)+'\n' )
        print( Zeros.__results_table(CTF, poly, win, xlim, ylim, defocus, Cs, results) )
        return

    # Make a table of results from fitting parameters.
    def __results_table( CTF, poly, win, xlim, ylim, defocus, Cs, results ):
        if ( Cs == None ):
            Cs = 'N/A'
        if ( defocus == None ):
            defocus = 'N/A'
        string = \
        '\n---------------'+\
        '\nFitting results'+\
        '\n---------------'+\
        '\nDefocus (nm): ' + str(defocus) +\
        '\nSpherical aberration (mm): ' + str(Cs) +\
        '\n----------------------'+\
        '\nRadial profile'+\
        '\n----------------------'+\
        '\nMaximum frequency (nm-1): ' + str(CTF.max_freq_inscribed) +\
        '\nFitted range x (nm-1): ' + str(xlim[0]) + ' - ' + str(xlim[1]) +\
        '\nFitted range y:'+ str(ylim[0]) + ' - ' + str(ylim[1]) +\
        '\n------------------------'+\
        '\nSavitksy-Golay smoothing' +\
        '\n------------------------'+\
        '\nPolynomial=' + str(poly) +\
        '\nWindow=' + str(win) +\
        '\n---------------'+\
        '\nDetected minima' +\
        '\n---------------'+\
        '\n|------------------|-------|'+\
        '\n| Frequency (nm-1) | Index |'+\
        '\n|------------------|-------|' 
        for n, m in zip(CTF.frequency[CTF.minima], CTF.indicies_min):
            space1 = ' '*(9-len(str(n)))
            space2 = ' '*(4-len(str(m)))
            string+='\n|'+space1+str(n)+'         |'+space2+str(m)+'   |'
        string+='\n|------------------|-------|' 
        return string
    