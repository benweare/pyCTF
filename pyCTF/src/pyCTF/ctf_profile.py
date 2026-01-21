import numpy as np

import scipy
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from pyCTF.misc import baseline_als

class Profile:
    '''
    Class for creating and manipulating line profiles of arrays.

    Attributes
    ----------
    radius : float, optional
        Maximum frequency in profile.

    Methods
    -------
    radial_profile()
        Generate line profile of square 2D array.
    crop_frequency()
        Crop profile to frequency range.
    remove_baseline()
        Measure and subtract baseline from profile.
    smooth_profile()
        Savitsky-Golay smoothing.
    plot_radial_profile()
        Plot with matplotlib.

    Notes
    -----
    This class is used to create and manipulate line profiles from square 2D
    arrays.
    '''
    def __init__( self ):
        self.radius = None
        return

    def radial_profile( data, centX, centY ):
        '''
        Create radial profile of 2D array.

        Parameters
        ----------
        data : array
            Input, square 2D array.
        centX : int
            Centre of array in x-axis.
        centY : int
            Centre of array in y-axis.

        Returns
        -------
        radialprofile :  array
            Output, 1D array.
        bins : int
            Length of radial profile.

        Notes
        -----
        Based on example from Stack Exchange:
        https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
        If used with iradius generated from a 2D array, the output is
        the frequency for the intensity radial profile. 
        '''
        y, x = np.indices(( data.shape ))
        r = np.sqrt((x - centX)**2 + (y - centY)**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        bins = len(nr)
        return radialprofile, bins

    def crop_frequency( data, freq, f_limits ):
        '''
        Crop profile to a range.

        Parameters
        ----------
        data : array
            Array to crop.
        freq : array
            Frequency array.
        f_limits : array
            Array as [x, y] to give cropping limits.

        Returns
        -------
        cropped_freq : array
        cropped_prof : array

        Warnings
        --------
        Method will not work if passed None type.
        Both data and freq must be the same length. 

        Notes
        -----
        Crops array data to the start and end points given in f_limits, 
        by comparing those limits to the array freq to find the element at 
        at which to crop both arrays. 
        '''
        # catch None being passed to cropping
        if f_limits[0] == None:
            f_limits[0] = data[0]
        if f_limits[1] == None:
            f_limits[1] = data[-1]
        range_low = np.argwhere( freq <= f_limits[0] )
        range_high = np.argwhere( freq >= f_limits[1] )
        cropped_freq = freq[range_low[-1,0]:range_high[1,0]] #freq[range_low[-1, 0]:range_high[1, 0]]
        cropped_prof = np.zeros( np.size( cropped_freq ) )
        cropped_prof = data[range_low[-1, 0]:range_high[1, 0]]
        return cropped_freq, cropped_prof
    
    # calls baseline_als and stores the result in class
    def remove_baseline( data ):
        '''
        Remove baseline of profile.

        Parameters
        ----------
        data : array

        Returns
        -------
        baseline : array

        Notes
        -----
        Calls misc.baseline_als() with preset values for lam and p 
        that should work for most data.
        '''
        baseline = baseline_als( data, 100, 0.0001 )
        return baseline

    def smooth_profile( data, window, polynomial, **kwargs ):
        '''
        Savitsky-Golay smoothing.

        Parameters
        ----------
        data : array
        window : int
            Size of smoothing window.
        polynomial : int
            Order of smoothing polynomial.
        axis : int, optional
        deriv : int, optional
        mode : string, optional

        Returns
        -------
        smoothed : array

        Notes
        -----
        Calls scipy.signal.savgol_filter() with preset values for
        axis, deriv, and mode that should work well for most data.
        '''
        axis = kwargs.get( 'axis', -1 )
        deriv = kwargs.get( 'deriv', 0 )
        mode = kwargs.get( 'mode', 'interp' )
        smoothed = savgol_filter( data,
                                window,
                                polynomial,
                                axis=-1,
                                deriv=0,
                                mode='interp')
        return smoothed
    
    def plot_radial_profile( prof, freq ):
        '''
        Matplotlib plot of profile.

        Parameters
        ----------
        prof : array
            y-axis data.
        freq : array
            x-axis data.
        '''
        fig, ax = plt.subplots(1, 1)
        ax.set_box_aspect(1)
        ax.set_title("Radial profile")
        ax.set_ylabel('Intensity', fontsize = 16)
        ax.set_xlabel('Frequency / nm-1', fontsize = 16)
        ax.plot( freq, prof, label='profile' )
        return