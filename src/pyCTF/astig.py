'''
A class to measure twofold astigmatism in CTFs.
'''

import numpy as np 
import matplotlib.pyplot as plt
import scipy

import skimage
from skimage.transform import warp_polar
from skimage.filters import gaussian

from pyCTF.misc import LineProfiles
from pyCTF.misc import ZerosData
from pyCTF.misc import make_scalebar
from pyCTF.misc import composite_image

from pyCTF.profile import Profile


def astig_magnitude( Image, defocus_guess, **kwargs ):
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
    from pyCTF.astig import Astig

    phi = kwargs.get( 'phi', Image.amax )
    astig_max = kwargs.get( 'max_val', 1000 )
    slices = kwargs.get( 'slices', 11 )
    
    # Multiply max freq by 2 as we need the diameter not the radius.
    CTF2D = CTFSimulation2D( Image.max_freq_inscribed*2,
                            int(Image.length),
                            Image.kV, -500)
    CTF2D.scale = Image.scale
    CTF2D.defocus = defocus_guess * 1e-9
    CTF2D.phi = np.deg2rad( phi )
    CTF2D.update()

    vals, a, polar_list = Astig.magnitude_measure( Image.image, 
                                                        slices, 
                                                        astig_max,
                                                        CTF2D,
                                                        radius=int(np.round((Image.length/2))) )
    x, y = Astig.__find_astig_defocus( vals, a )
    
    CTF2D.C12a = x[0] *1e-9
    CTF2D.C12b = y[0] *1e-9
    CTF2D.update()
    
    Astig.plot_results( vals, slices, a, CTF2D )
    print( str(x[0]) + ', ' + str(y[0]) )
    return


# Methods for astigmatism measurement.
def astig_angle( Image ):
    '''
    Measure angle of twofold astigmatism in CTF.

    Notes
    -----
    Wrapper to streamline measuring astigmatism to a single call.
    '''
    # check for profiles
    Image.radial_profile, _ = Profile.radial_profile( Image.image, Image.centX, Image.centY )
    Image.frequency, _ = Profile.radial_profile( Image.iradius, Image.centX, Image.centY )
    Astig.measure_angle( Image )
    return


def print_astig_results( Image ):
    '''
    Show outcome of twofold astigmatism measurement.

    Notes
    -----
    Wrapper around Zeros.plotCsFigure() and Zeros.printResults().
    '''
    Zeros.plotCsFigure( Image.cropped_frequency, 
                        Image.smoothed_profile, 
                        Image.minima, 
                        Image.x_min, 
                        Image.y_min, 
                        Image.results,
                        Image.cropped_frequency,
                        (Image.cropped_profile-Image.baseline) )
    
    Zeros.printResults( Image.defocus, 
                            Image.Cs, 
                            Image.results )
    return


class Astig( LineProfiles ):
    '''
    Class for measuring twofold astigmatism in the CTF.

    Attributes
    ----------
    CTF : class
        CTF_image class.
    radius : float
    masked : array
    fmax : float
        Maximum defocus.
    fmin : float
        Minimum defocus.
    amin : float
        Minimum angle.
    amax : float
        Maximum angle.

    Methods
    -------
    measure_angle( )
    correlate_angle( )
    calc_angles( )
    make_data( slices, a, b, simCTF, radius )
    magnitude_correlate( warped, polar )
    magnitude_measure( image, slices, max_val, CTF2D, **kwargs )
    find_astig_defocus( vals, a )
    apply_limit( vals, limits )
    plot_results( vals, slices, a, CTF2D )
    plot_angles( )
    print_all( )

    Notes
    -----
    Used almost exclusively in conjunction with the CTF_image class.

    For a mathematical description of astigmatism, see the literautre.
    '''


    def measure_angle( Image ):
        '''
        Returns angle of astigmatism.

        Notes
        -----
        Uses the literature method to find the angle of astigmatism in a CTF.
        Uses skimage.transform.warp_polar() to convert from cartesian to polar
        reprentation of CTF, then searches for maximum displacement via
        cross-correlation using twofoldAstigmatism.correlate_angle(). 
        '''
        #from skimage. transform import warp_polar
        #from skimage.filters import gaussian
        polar = warp_polar( Image, ( Image.centX,Image.centY ),
            radius = Image.length/2 )
        polar = gaussian( polar, sigma=1.5 )
        polar = polar[ :, 40:350 ]
        Image.amax, Image.correlation, Image.maximum, Image.minimum = Astig.correlate_angle( Image )
        if (Image.amax > 360 or Image.amax < 0):
            print( 'angleError: returned angle was greater than 360 or less than 0.' )
            print( 'Found angle (degrees): ' + str(self.amax) )
        if ( Image.amax >= 180 ):
            Image.amax = Image.amax - 180
        Image.amin = Image.amax - 90
        if ( Image.amin < 0 ):
            Image.amin = Image.amax + 90
        if ( Image.amin > 360 ):
            Image.amin = Image.amax - 90
        return


    def correlate_angle( Image ):
        '''
        Autocorrelation of image to find astigmatism angle.

        Notes
        -----
        Works in conjunction with twofoldAstigmatism.measure_angle(). 
        Uses scipy.signal.correlate to correlate the polar form of the CTF with 
        it's mirror image, then uses numpy.where() to find the maximum and minima 
        of the cross-correlation.
        '''
        from scipy.signal import correlate
        output = correlate( Image.polar, np.flip( Image.polar, 0 ), mode='same' )
        maximum = np.where( output == output.max() )
        minimum = np.where( output == output.min() )
        angle = maximum[0]*(np.size( Image.polar[1] ) / 360 )
        return angle, output, maximum, minimum


    def calc_angles( Image ):
        '''
        Calculate values to draw lines on an image.

        Notes
        -----
        Used by twofoldAstigmatism.plot_results().
        '''
        # line coordinates
        l = 500 
        a1 = Image.amax
        a2   = Image.amin
        x1 = Image.centX + l * np.sin(np.radians( a1 ))
        y1 = Image.centY - l * np.cos(np.radians( a1 ))
        x2 = Image.centX + l * np.sin(np.radians( a2 ))
        y2 = Image.centY - l * np.cos(np.radians( a2 ))
        return x1[0], y1[0], x2[0], y2[0]


    ### methods to find astigmatism magnitude with cross-correlation
    def __make_data( slices, a, b, simCTF, radius ):
        '''
        Simulate CTFs with a range of twofold astigmatism.

        Parameters
        ----------
        slices : int
        a : float
            Maximum astigmatism value.
        b: array
            Array of astigmatism values.
        simCTF : class
            CTF_simulation_2D class.
        radius : float

        Warnings
        --------
        The speed of this method is inversely proportional to the number of 
        simulations to perform, and the size of each simulated array. 

        Returns
        -------
        polar : array

        Notes
        -----
        Creates an array containing simulated 2D CTFs in polar form, for use
        with twofoldAstigmatism.magnitude_correlate() to measure the magnitude
        of the astigmatism present in the experimental CTF.

        Simulations vary the minimum defocus due to astigmatism (b) with
        respect to the maximum defocus due to astigmatism (a), skipping values
        where b is greater than a to avoid redundancy and increase speed. 

        If a = 0 nm, it is set to 0.01 nm to prevent a divide-by-zero error.
        '''
        from skimage. transform import warp_polar
        polar = np.zeros( (slices, 360, radius)  ) # here: 182 for full
        if a == 0:
            a = 0.01 #smallest a very small, to avoid dividing by zero
        for n in range( slices ):
            if a >= b[n] and a != b[n]:
                simCTF.C12a = a*1e-9
                simCTF.C12b = b[n]*1e-9
                simCTF.update()
                polar[n] = warp_polar( simCTF.square_CTF[:, :], radius=radius ) # full
        return polar
    

    # See CTFFIND4 paper for method used here.
    def __magnitude_correlate( warped, polar ):
        '''
        Pearson's correlation coeffcient to determine astigmatism magnitude. 

        Parameters
        ----------
        warped : array
        polar : array

        Returns
        -------
        val : array
            Array of correlation coeffcients.

        Notes
        -----
        Based on the literature methods (CTFFind 4).

        Uses Pearson's correlation coeffcient (Scipy) to determine how well
        simulated CTFs match the experimental CTF. 
        '''
        from scipy.stats import pearsonr
        val = np.zeros( (np.size( polar, 0)) )
        for n in range( np.size( polar, 0 ) ):
            if ( np.sum(polar[n, :, :]) != 0):
                val[n], _ = pearsonr( warped, polar[n, :, :], axis=None )
            else:
                val[n] = None
        return val
    

    def magnitude_measure( image, slices, max_val, CTF2D, **kwargs ):
        '''
        Wrapper to measure magnitude of astigmatism.

        Parameters
        ----------
        image : array
            Exeperimental CTF as an array.
        slices : int
        max_val : float
        CTF2D : class
            simulate_2D_CTF class. 
        radius : float, optional
            Value to crop images to.

        Returns
        -------
        vals : array
        a : float 
        polar_list : array

        Warnings
        --------
        The speed of this method is limited by the number of simulations to peform, and 
        the size of each simulated image. 

        Notes
        -----
        This method uses several methods in the twofoldAstigmatism class to measure the 
        magnitude of the twofoldAstigmatism in an experimental CTF, by correlation with 
        simulated stigmated CTFs. 
        '''
        radius = kwargs.get( 'radius', int(( np.round( np.size(image, 0)/2) ) ))
        #min_val=0
        from skimage. transform import warp_polar
        vals = np.zeros( (slices, slices) )
        polar_list = []
        a = np.linspace( 0, max_val, slices )
        b = np.linspace( 0, max_val, slices )
        warped = warp_polar( image, radius=radius )
        warped = warped[:90, :]
        # change logic so don't have to evaluate if statement every time? 
        for n in range( slices ):#range(np.size(a, 0))
            polar = Astig.__make_data( slices, a[n], b, CTF2D, radius )
            polar = polar[:, :90, :]
            if ( n==0 ):
                polar_list.insert( 0, polar )
            else:
                polar_list.append( polar )
            vals[n, :] = Astig.__magnitude_correlate( warped, polar )
        return vals, a, polar_list


    # Other methods.
    def __find_astig_defocus( vals, a ):
        '''
        Returns the magnitude of astigmatism.

        Parameters
        ----------
        vals : array
        a : float

        Returns
        -------
        a[x] : float
        a[y] : float

        Notes
        -----
        '''
        x, y = np.where( vals == np.nanmax(vals) )
        return a[x], a[y]
    

    # Under development, see CTFFIND4 paper for method of scoring.
    def __apply_limit( vals, limits ):
        vals_adjusted = vals - ((a[x] - a[y])**2 / (2*( limits**2)*256))
        return vals_adjusted


    # Plotting functions.
    def plot_results( Image, vals, slices, a, CTF2D ):
        '''
        Plot results of CTF astigmatism determination.

        Parameters
        ----------
        vals : array
        slices : int
        a : array
        CTF2D : class
            simulate_2D_CTF class.

        Notes
        -----
        Uses Matplotlib to generate figures.
        '''
        fig, axs = plt.subplots( 1, 2, figsize=(10, 10) )
        a = np.round(a)
        axs[0].matshow( vals )
        axs[0].set_xticks(range(slices))
        axs[0].set_yticks(range(slices))
        a = np.round(a, decimals=1)
        try:
            axs[0].set_xticklabels(a)
            axs[0].set_yticklabels(a)
        except:
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
        axs[0].set_xlabel('C12b')
        axs[0].set_ylabel('C12a')
        composite = composite_image( Image.image, CTF2D.square_CTF,
            int(np.round(Image.length/2)) )
        axs[1].matshow( composite )
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        scalebar = make_scalebar( 1, Image.scale, axs[1] )
        axs[1].add_artist(scalebar)
        #draw lines on image
        x1, y1, x2, y2 = self.calc_angles( )
        axs[1].axline(( Image.centX, Image.centY), (x1, y1), color='red')
        axs[1].axline(( Image.centX, Image.centY), (x2, y2), color='orange')
        axs[1].set_xlim([0, Image.width])
        axs[1].set_ylim([Image.length, 0])
        return


    def plot_angles( Image ):
        '''
        Plot results of astigmatism angle determination.

        Notes
        -----
        Acts on twofoldAstigmatism class, uses Matplotlib.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(8, 8))
        axs[0].matshow( Image.polar )
        axs[0].hlines(Image.amax, 0, len(Image.polar), color='orange',
            label='Minimum defocus')
        axs[0].hlines(Image.amin, 0, len(Image.polar), color='red',
            label='Maximum defocus')
        axs[0].set_xlim([0, len(Image.polar[0])])
        #axs[0].set_ylim([0, len(self.polar[1])])
        axs[1].matshow( Image.correlation )
        axs[1].plot( Image.maximum[1], Image.maximum[0], 'x', color='red',
            label='Maximum' )
        titles = ['Polar image', 'Autocorrelation']
        a = [ axs[0], axs[1] ]
        n = 0
        for ax in a:
            ax.set_title( titles[n] )
            ax.set_yticks([])
            ax.set_xticks([])
            n = n+1
        #fig.legend( loc='' )
        try:
            scalebar = make_scalebar( 0.5, Image.scale, axs[0] )
            axs[0].add_artist(scalebar)
        except:
            print('Error: could not add scalebar to image.')
        axs[0].set_ylabel('Angle / degrees', fontsize = 12)
        axs[0].set_xlabel('Frequency / $nm^{-1}$', fontsize = 12)
        axs[0].legend(loc='lower right')
        axs[1].legend(loc='lower right')
        return


#    # print all simulated CTF
#    def print_all( self ):
#        '''
#        Display all simulated CTFs in a figure.
#
#        Warnings
#        --------
#        For large number of simulated CTFs, the size of each figure axis is
#        very small.
#        '''
#        fig, axs = plt.subplots( slices, slices )
#        for n in range( slices ):
#            for m in range( slices ):
#                axs[n, m].matshow( polar_list[n][m, :, :] )
#                string = str(n) + ", " + str(m)
#                axs[n, m].set_title( string )
#                axs[n, m].set_xticks([])
#                axs[n, m].set_yticks([])
#        return