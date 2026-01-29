'''
Module for Fourier-space methods.
'''

import numpy as np 
import matplotlib.pyplot as plt

import skimage
from skimage import io
from skimage.filters import gaussian

from pyCTF.misc import find_iradius_itheta
from pyCTF.misc import normalise_data_range

from pyCTF.ctf_profile import Profile

class Fourier:
    '''
    Methods for performing Fourier-space operations on arrays.

    Methods
    -------
    imfft( image )
    inv_imfft( image )
    log_mod( image )
    fft_stack( stack )
    fft3d( stack )
    plot_fft3d( data )
    crop( data, width, **kwargs )
    remove_bckg( image, rstart1, rstart2 )
    remove_bckg_stack( stack, rstart1, rstart2 )
    profile_fft_stack( stack )
    show_fft( image )
    through_focus( stack, **kwargs )
    import_stack( string )
    process_CTF( image )
    measure_arcs( image, width )
    plot_arcs( image, output )

    Notes
    -----
    Class containing method for performing Fourier-space operations on arrays,
    using numpy.fft class. Contains wrappers for basic operations such as
    performing a FFT and moving 0-frequency terms to center of output array,
    and more complex functions for CTF data processing such as Fourer-space
    background removal.


    3D arrays are referred to as "stacks", with constituent 2D arrays referred
    to as "slices", following image processing nomenclature. Stacks use the
    following coordinate convention: [[x:, y:], z], where for each integer
    value of z is associated with a slice [x:, y:] that corresponds to an
    image.

    The terms DC (direct-current), zeroth order, and 0-order frequency may be
    used interchangably in this class's documentation, sorry in advance. 
    '''
    def __init__( self ):
        from numpy.fft import fft2
        from numpy.fft import ifft2
        from numpy.fft import fftshift
        return

    def imfft( image ):
        '''
        Fast Fourier transform of a square array.

        Parameters
        ----------
        image : array
            Real or interger valued array.

        Returns
        -------
        FT : array
            Complex-valued array.

        Notes
        -----
        Moves DC frequencies to centre of output array.
        '''
        FT = np.fft.fft2( image )
        FT = np.fft.fftshift( FT )
        return FT

    def inv_imfft( image ):
        '''
        Inverse fast Fourier transform of a square array.

        Parameters
        ----------
        image : array
            Complex-valued valued array.

        Returns
        -------
        imfft : array
            Real or integer valued array.
        '''
        imfft = np.fft.ifft2( imfft )
        return imfft

    def log_mod( image ):
        '''
        Log-modulus of array.

        Parameters
        ----------
        image : array
            Complex-valued valued array.

        Returns
        -------
        logmod : array
            Real or integer valued array.

        Notes
        -----
        Takes a complex-valued array and returns the natural logarithim 
        of it's modulus, allowing it to be displayed e.g. with Matplotlib.
        '''
        logmod = np.log( np.abs( image ) )
        return logmod

    # fix counting stack length
    def fft_stack( stack ):
        '''
        2D FFT on each slice in a stack. 

        Parameters
        ----------
        stack : array
            Real or integer valued array, where each slice corresponds to an
            image.

        Returns
        -------
        output : array
            Complex-valued array, where each slice is the FFT of the
            correpsonding slice in stack.

        Notes
        -----
        Stacks use the following coordinate convention: [[x, y], z], where for
        each integer value of z is associated with an image [x, y]. 
        '''
        output = np.array( stack, complex )
        for n in range( 0, np.size(stack, 2) ):
            output[:, :, n] = Fourier.imfft( stack[:, :, n] )
        return output

    def fft3d( stack ):
        '''
        3DFFT of a stack.

        Parameters
        ----------
        stack : array
            Real or integer valued array, where each slice corresponds to an
            image.

        Returns
        -------
        output : array
            Complex-valued array.

        Notes
        -----
        Performs the 3D FFT of an image stack, with the zeroth order
        frequencies shifted to the centre. 
        '''
        output = np.fft.fftn( stack, axes=(0, 1, 2) )
        output = np.fft.fftshift( output )
        #output = np.fft.fftshift( output, axes=(0, 2) )
        return output

    # plot views of three axes of 3D FFT
    # unfinished
    def plot_3d_fft( data ):
        '''
        Plot 3D FFT.

        Notes
        -----
        Unfinished. 
        '''
        fig, axs = plt.subplots( 1, 2 )
        axs[0].matshow( data[:, :, 0] )
        # rotate
        axs[1].matshow( data )
        return

    # 
    def crop( image, width, **kwargs ):
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

    # redundant with method in CTF image class, but more general
    def remove_bckg( image, rstart1, rstart2 ):
        '''
        Remove background from CTF via Fourier methods.

        Parameters
        ----------
        image : array
        rstart1 : float
        rstart2 : float

        Returns
        -------
        image : array

        Notes
        -----
        Redundant with method in CTF_image class, but more generalised. 
        Background removal follows literature method. 
        '''
        # low frequency
        imfft = np.fft.fft2( image )
        imfft = np.fft.fftshift( imfft )
        iradius, _ = find_iradius_itheta( imfft, 1 )
        n = range(0, np.size(iradius,0))
        m = range(0, np.size(iradius,1))
        for i in n:
            for j in m:
                if iradius[i,j] >= rstart1:
                    imfft[i,j] = 0
        imfft = np.fft.ifft2( imfft )
        LF_bkg = np.abs( imfft )
        image = image - np.abs( imfft )
        # high frequency
        imfft = np.abs( image ) #natural log
        imfft = np.fft.fft2( imfft)
        imfft = np.fft.fftshift( imfft )
        iradius,_ = find_iradius_itheta( imfft, 1 )
        n = range(0, np.size(iradius,0))
        m = range(0, np.size(iradius,1))
        for i in n:
            for j in m:
                if iradius[i,j] >= rstart2:
                    imfft[i,j] = 0
        imfft = np.exp( np.abs(np.fft.ifft2( imfft )) )
        E_bkg = imfft
        image = image / E_bkg
        del( imfft )
        del( iradius )
        return image, LF_bkg, E_bkg

    def remove_bckg_stack( stack, rstart1, rstart2 ):
        '''
        Remove background from each slice in a stack of Fourier transforms.

        Parameters
        ----------
        stack : array
        rstart1 : float
        rstart2 : float

        Returns
        -------
        output : array

        Notes
        -----
        Wrapper around remove_bckg() allowing each slice of a stack to be
        processed and outputted to a new stack.
        '''
        out = np.array( stack )
        for n in range( np.size( stack, 2) ):
            out[:, :, n], _, _ = Fourier.remove_bckg( stack[:, :, n],
                                                    rstart2, 
                                                    rstart2 )
        return out

    # radial profiles of FFT stack
    def profile_fft_stack( stack ):
        '''
        Remove background from each slice in a stack.

        Parameters
        ----------
        stack : array
            3D array.

        Returns
        -------
        output : array
            1D array.

        Warnings
        --------
        May not behave as expected if supplied complex data. 

        Notes
        -----
        Wrapper around profile.radial_profile(), allowing radial profile of 
        each slice in a stack to be calculated and outputted to a new stack of
        radial profiles. 
        '''
        centX = len(stack[0])/2
        centY = len(stack[1])/2
        # get length of radial profile
        out, _ = Profile.radial_profile( stack[:,:,0], centX, centY )
        output = np.zeros( [ np.size(out), np.size(stack, 2)] )
        for n in range( np.size(stack, 2) ):
            output[:, n], _ = Profile.radial_profile( stack[:, :, n], 
                                                    centX, 
                                                    centY )
        return output

    def show_fft( image ):
        '''
        Display a Fourier transform as an image.

        Parameters
        ----------
        image : array

        Raises
        ------
        error
            Input array was complex-valued.

        Warnings
        --------
        Assumes the Foruier transform has been converted to a real or integer
        valued array.
        '''
        fig, ax = plt.subplots( 1 )
        ax.matshow( image )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # FT 2D stack and get radial profile
    # update default radii
    def through_focus( stack, **kwargs ):
        '''
        Process a through-focus series

        Parameters
        ----------
        stack : array
            Real or integer valued array.
        width : int, optional
            Pixel diameter for center crop.
        r1 : int, optional
        r2 : int, optional
        verbose : string, optional
            True to toggle console output.

        Returns
        -------
        FFT : array
            Real or integer valued array.
        prof : array
            Line profiles of FFT.

        Notes
        -----
        Wrapper around various Fourier class functions, to process a stack of 
        images and give a stack of CTFs and their radial profiles. For
        individual methods, see: FFTstack(), log_mod(), crop(),
        remove_bckg_stack(), profile_FFT_stack().
        '''
        width = kwargs.get( 'width', None )
        r1 = kwargs.get( 'r1', 10 )
        r2 = kwargs.get( 'r2', 10 )
        verbose = kwargs.get( 'verbose', False )
        if (verbose == True):
            print( '\nStarting processing through-focus series.' )
            print( 'Performing FFT of stack.' )
        FFT = Fourier.fft_stack( stack )
        if (verbose == True):
            print( 'Coverting stack to real data.' )
        FFT = Fourier.log_mod( FFT )
        if ( width != None ):
            if (verbose == True):
                print( 'Center cropping data to range.' )
            FFT = Fourier.crop( FFT, width )
        if (verbose == True):
            print( 'Removing background from CTF.' )
        FFT = Fourier.remove_bckg_stack( FFT, r1, r2 )
        if (verbose == True):
            print( 'Measuring line profile of stack.' )
        prof = Fourier.profile_fft_stack( FFT )
        if (verbose == True):
            print( 'Done.' )
        return FFT, prof

    # import TIFF stack as a stack
    def import_stack( string ):
        '''
        Import uncompressed TIFFs as a stack.

        Parameters
        ----------
        string : string

        Returns
        -------
        stack : array

        Raises
        ------
        error
            Tried to import a compressed TIF.

        Notes
        -----
        Each slice in the array corresponds to an individual image.

        Importing uses io.imread, so only works on uncompressed TIFFs. For
        example, TIFFs exported from ImageJ are often uncompressed, while
        those exported from DigitalMicrograph are often compressed. 
        '''
        stack = np.array( io.imread( string ) )
        # rearrange to convention defined above
        stack = np.moveaxis( stack, 0, 2 )
        return stack

    # add kwargs for variable in last two methods
    def process_CTF( image, **kwargs ):
        '''
        Process a single image to get the CTF.

        Parameters
        ----------
        image : array
        width : int, optional
            Number of pixels to crop image to.
        rs1 : int, optional
        rs2 : int, optional

        Returns
        -------
        FT : array

        Notes
        -----
        Wrapper around various functions in Foruier class, for method details 
        see: imFFT(), log_mod(), crop(), remove_bckg(). 

        Processes an image to get the CTF, using default values that may work
        well for some datasets. 
        '''
        width = kwargs.get( 'width', 300 )
        rs1 = kwargs.get( 'rs1', 20 )
        rs2 = kwargs.get( 'rs2', 20 )
        FT = Fourier.imfft( image )
        FT = Fourier.log_mod( FT )
        FT = Fourier.crop( FT, width )
        FT, _, _ = Fourier.remove_bckg( FT, rs1, rs2 )
        return FT

    def measure_arcs( image, width ):
        '''
        Measure the arcs in the 3D Fourier transform.

        Parameters
        ----------
        image : array
            Numpy array.
        width : float

        Returns
        -------
        output : array
        filtered : array

        Notes
        -----
        See literature for background.
        '''
        # crop to half size
        end = int((np.size(image,0)/2))
        middle = int( np.size(image,1)/2 )
        filtered = image[ 0:end, :]
        # gaussian blur
        from skimage.filters import gaussian
        filtered = skimage.filters.gaussian( filtered, sigma=1.0 )
        output = np.zeros( (np.size(filtered[:, 1]), 3) )
        # left arc
        output[:, 0] = np.argmax( filtered[:, 0:(middle - width) ], axis=1 )
        # right arc
        output[:, 1] = np.argmax( filtered[:, (middle + width): ], axis=1 ) 
        + middle
        output[:, 2] = range(np.size(filtered[:, 1]))
        return output, filtered

    def plot_arcs( image, output ):
        '''
        Plot detected arcs.

        Parameters
        ----------
        image : array
        output : array

        Notes
        -----
        Uses Returns from Fourier.measure_arcs().
        See literature for background.
        '''
        # Can see where arc is not detected when differece spikes down.
        fig, axs = plt.subplots(1, 2, figsize=(8, 8))
        axs[0].matshow( image )
        axs[0].plot( output[:, 0], output[:, 2], 'x', color='red' )
        axs[0].plot( output[:, 1], output[:, 2], 'x', color='orange' )
        
        output[:, 0] = normalise_data_range( output[:, 0] )
        output[:, 1] = normalise_data_range( output[:, 1] )
    
        axs[1].hlines(1, 0, 100, color='k', linestyle='--', alpha=0.7 )
        axs[1].hlines(0, 0, 100, color='k', linestyle='--', alpha=0.7 )
        axs[1].plot( np.diff( output[:, 0] )
            +1.0, alpha=1.0, 
            label='left', color='red' )
        axs[1].plot( np.diff( output[:, 1] ),
            alpha=1.0,
            label='right',
            color='orange' )
        
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_yticks([])
        #axs[1].legend()
        axs[1].legend()
    
        axs[0].set_title( 'arcs' )
        axs[1].set_title( 'difference' )
        #fig.savefig( '80kV_arcs.png', dpi='figure', format='png' )
        return