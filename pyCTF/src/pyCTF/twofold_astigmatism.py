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


# class for measuring astigmatism in CTF_image.image
class twofoldAstigmatism( LineProfiles ):
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
    plot_3D( )
    print_all( )
    mask( )

    Notes
    -----
    Used almost exclusively in conjunction with the CTF_image class.

    For a mathematical description of astigmatism, see the literautre.
    '''
    def __init__( self, CTF ):
        '''
        Parameters
        ----------
        CTF : class
            CTF_image class.
        '''
        self.CTF = CTF
        LineProfiles.__init__( self )
        ZerosData.__init__( self )
        self.radius = None
        self.masked = None
        # defocus of astigmatism
        self.fmax = None
        fmin = None
        # angle of astigmatism
        self.amin = None
        self.amax = None
        return


    ### methods to measure astigmatism angle ###
    def measure_angle( self ):
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
        self.polar = warp_polar( self.CTF.image, ( self.CTF.centX,
                                                self.CTF.centY ),\
            radius = self.CTF.length/2 )
        self.polar = gaussian( self.polar, sigma=1.5 )
        self.polar = self.polar[ :, 40:350 ]
        self.amax, self.correlation, self.maximum, self.minimum = self.correlate_angle()
        if (self.amax > 360 or self.amax < 0):
            print( 'angleError: returned angle was greater than 360 or less than 0.' )
            print( 'Found angle (degrees): ' + str(self.amax) )
        if ( self.amax >= 180 ):
            self.amax = self.amax - 180
        self.amin = self.amax - 90
        if ( self.amin < 0 ):
            self.amin = self.amax + 90
        if ( self.amin > 360 ):
            self.amin = self.amax - 90
        return


    # autocorrelate to find angle
    def correlate_angle( self ):
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
        output = correlate( self.polar, np.flip( self.polar, 0 ), mode='same' )
        maximum = np.where( output == output.max() )
        minimum = np.where( output == output.min() )
        angle = maximum[0]*(np.size( self.polar[1] ) / 360 )
        return angle, output, maximum, minimum


    def calc_angles( self ):
        '''
        Calculate values to draw lines on an image.

        Notes
        -----
        Used by twofoldAstigmatism.plot_results().
        '''
        # line coordinates
        l = 500 
        a1 = self.CTF.astig.amax
        a2   = self.CTF.astig.amin
        x1 = self.CTF.centX + l * np.sin(np.radians( a1 ))
        y1 = self.CTF.centY - l * np.cos(np.radians( a1 ))
        x2 = self.CTF.centX + l * np.sin(np.radians( a2 ))
        y2 = self.CTF.centY - l * np.cos(np.radians( a2 ))
        return x1[0], y1[0], x2[0], y2[0]


    ### methods to find astigmatism magnitude with cross-correlation
    # try to speed this up by only making the needed part of the CTF simulation? 
    def make_data( self, slices, a, b, simCTF, radius ):
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
        Creates an array containing simulated 2D CTFs in polar form, for use with 
        twofoldAstigmatism.magnitude_correlate() to measure the magnitude of the 
        astigmatism present in the experimental CTF.

        Simulations vary the minimum defocus due to astigmatism (b) with respect to the 
        maximum defocus due to astigmatism (a), skipping values where b is greater than 
        a to avoid redundancy and increase speed. 

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
                polar[n] = warp_polar( simCTF.squareCTF[:, :], radius=radius ) # full
        return polar
    

    # CTFfind 4
    def magnitude_correlate( self, warped, polar ):
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

        Uses Pearson's correlation coeffcient (Scipy) to determine how well simulated CTFs 
        match the experimental CTF. 
        '''
        from scipy.stats import pearsonr
        val = np.zeros( (np.size( polar, 0)) )
        for n in range( np.size( polar, 0 ) ):
            if ( np.sum(polar[n, :, :]) != 0):
                val[n], _ = pearsonr( warped, polar[n, :, :], axis=None )
            else:
                val[n] = None
        return val
    

    def magnitude_measure( self, image, slices, max_val, CTF2D, **kwargs ):
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
            polar = self.make_data( slices, a[n], b, CTF2D, radius )
            polar = polar[:, :90, :]
            if ( n==0 ):
                polar_list.insert( 0, polar )
            else:
                polar_list.append( polar )
            vals[n, :] = self.magnitude_correlate( warped, polar )
        return vals, a, polar_list


    ## other methods
    def find_astig_defocus( self, vals, a ):
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
    

    def _apply_limit( vals, limits ):
        '''
        Adjust scoring to favour lower astigmatism values.

        Parameters
        ----------
        vals : array
        limits : float

        Returns
        -------
        vals_adjusted : array

        Notes
        -----
        Under development, based on literature example (CTFFind 4).
        '''
        vals_adjusted = vals - ((a[x] - a[y])**2 / (2*( limits**2)*256))
        return vals_adjusted


    ## plotting functions ##
    def plot_results( self, vals, slices, a, CTF2D ):
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
        CTF = self.CTF
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
        composite = composite_image( CTF.image, CTF2D.squareCTF, int(np.round(CTF.length/2)) )
        axs[1].matshow( composite )
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        scalebar = make_scalebar( 1, CTF.scale, axs[1] )
        axs[1].add_artist(scalebar)
        #draw lines on image
        x1, y1, x2, y2 = self.calc_angles( )
        axs[1].axline(( CTF.centX, CTF.centY), (x1, y1), color='red')
        axs[1].axline(( CTF.centX, CTF.centY), (x2, y2), color='orange')
        axs[1].set_xlim([0, CTF.width])
        axs[1].set_ylim([CTF.length, 0])
        ### add ellipses at minima?
        #from matplotlib.patches import Ellipse
        #ellipse = Ellipse( (400, 400), 2.6/CTF.scale, 2/CTF.scale, angle=-60, fill=False, edgecolor='r' )
        #ax.add_patch( ellipse )
        #ax.plot( CTF.centX, CTF.centY, 'x', color='r' )
        return


    def plot_angles( self ):
        '''
        Plot results of astigmatism angle determination.

        Notes
        -----
        Acts on twofoldAstigmatism class, uses Matplotlib.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].matshow( self.polar )
        axs[0].hlines(self.amax, 0, len(self.polar), color='red', label='minimum defocus')
        axs[0].hlines(self.amin, 0, len(self.polar), color='orange', label='maximum defocus')
        axs[0].set_xlim([0, len(self.polar[0])])
        #axs[0].set_ylim([0, len(self.polar[1])])
        axs[1].matshow( self.correlation )
        axs[1].plot( self.maximum[1], self.maximum[0], 'x', color='red' )
        titles = ['polar image', 'autocorrelation']
        a = [ axs[0], axs[1] ]
        n = 0
        for ax in a:
            ax.set_title( titles[n] )
            ax.set_yticks([])
            ax.set_xticks([])
            n = n+1
        #fig.legend( loc='' )
        return


    #3D plots of astig angle search
    def plot_3D( self, vals, a ):
        '''
        Create a 3D plot of astigmatism angle search.
        '''
        X, Y = np.meshgrid(a, a)
        R = np.sqrt(X**2 + Y**2)
        Z = vals
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z, vmin=Z.min())
        return


    # print all simulated CTF
    def print_all( self ):
        '''
        Display all simulated CTFs in a figure.

        Warnings
        --------
        For large number of simulated CTFs, the size of each figure axis is
        very small.
        '''
        fig, axs = plt.subplots( slices, slices )
        for n in range( slices ):
            for m in range( slices ):
                axs[n, m].matshow( polar_list[n][m, :, :] )
                string = str(n) + ", " + str(m)
                axs[n, m].set_title( string )
                axs[n, m].set_xticks([])
                axs[n, m].set_yticks([])
        return
    
    ### not in use ###
    '''
    Masking function.

    Parameters
    ----------
    angle : float
        Starting value.
    width : float
        Angle segment.

    Notes
    -----
    Not currently in use. Creates of angle wedges.
    Used in conjection with CTF_image.astig_defocus().
    '''

    def _mask( self, angle, width ):
        no_sectors = 1
        N = 1
        interval = np.pi / no_sectors
        angle = angle - 180
        astart = np.deg2rad( angle - width )
        aend = np.deg2rad(angle + width )
        # mirror if angle is close to 360 or 0
        if (((angle - width) < -180)):
            astart = astart + np.pi
            aend = aend + np.pi
        if (((angle + width) > 180) ):
            astart = astart - np.pi
            aend = aend - np.pi
            
        mask = np.array( CTF.image )

        n = range(0, np.size( CTF.itheta, 0 ))
        m = range(0, np.size( CTF.itheta, 1 ))
        for i in n:
            for j in m:
                if CTF.itheta[i,j] >= astart and CTF.itheta[i,j] <= aend:
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0
        mask = np.flip( mask ) + mask
        self.masked = np.multiply( CTF.image, mask )
        # then do radial profile of mask, then get defocus of mask
        #then, should be ready to process zemlin tableaus
        return