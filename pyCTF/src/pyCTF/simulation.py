'''
Classes for simulating contrast transfer functions.

This module contains classes for simulating 2D and 1D CTFs.
'''

import numpy as np 
import matplotlib.pyplot as plt

from pyCTF.misc import kv_to_lamb
from pyCTF.misc import LensAberrations
from pyCTF.misc import LineProfiles


class CTFSimulation2D:
    '''
    Class for simulating 2D CTFs.

    Attributes
    ----------
    Cs : float
    C12a : float
    C12b : float
    phi : float
    beta : float
    image_size : float
    kV : float
    lamb : float
    flim : float
    cutoff : float
    defocus : float
    scale : float
    imageX : int
    imageY : int
    cent : array
    iradius : array
    itheta : array
    focal_spread : float
    CTF : array
    aperutre : array
    temporal : array
    spatial : array
    damped_CTF : array
    square_CTF : array

    Methods
    ------
    plot_ctf()
    update()
    print_aberrations()
    show_all()

    Notes
    -----
    flim is diameter, not radius.
    '''
    def __init__( self, flim, image_size, kV, defocus ):
        '''
        Parameters
        ----------
        flim : float
        image_size : int
        kV : float
        defocus : float
        '''
         ## lens aberrations ##
        LensAberrations.__init__( self )
        self.Cs = 1.6 * 1e-3
        self.Cc = 1.6 * 1e-3
        self.C12a = 0 * 1e-9 # minor defocus axis
        self.C12b = 0 * 1e-9 # major defocus axis
        self.phi = 0 
        self.beta =  0 * 1e-3
        LineProfiles.__init__( self )
        ## params ##
        self.image_size = image_size
        self.kV = kV
        self.lamb = kv_to_lamb( kV )
        self.flim = flim * 1e9
        self.cutoff = flim * 1e9
        self.defocus = defocus * 1e-9
        self.scale = self.image_size / self.flim
        ## images ##
        self.imageX = image_size
        self.imageY = image_size
        self.CTF = np.ones(( self.imageX, self.imageY ))
        self.cent = np.array([ self.CTF.shape[0] / 2.0, self.CTF.shape[0] ]) 
        self.iradius, self.itheta = self.__find_radial_distance( )
        self.focal_spread = 5.25
        ## simulations ##
        self.CTF = self.__simulate_2D_CTF()
        self.aperture = self.__aperture_function( )
        self.temporal = self.__temporal_coherence( )
        self.spatial= self.__spatial_coherence( )
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2
    

    # Find radial distance and angle for each pixel in the image.
    # Note, update to use the corresponding function in misc.
    def __find_radial_distance( self ):
        irow, icol = np.indices(self.CTF.shape)
        centX = irow - self.CTF.shape[0] / 2.0
        centY = icol - self.CTF.shape[1] / 2.0
        #distance from centre
        iradius = ((centX**2 + centY**2)**0.5) / self.scale
        #angle from centre
        itheta = np.arctan2(centX, centY)
        return iradius, itheta


    # Simulates 2D CTF.
    def __simulate_2D_CTF( self ):
        CTF = np.sin( (np.pi*self.defocus*self.lamb*(self.iradius**2) ) + 
                      (0.5*np.pi*self.Cs*(self.lamb**3)*(self.iradius**4)) +
                      np.pi*self.lamb*(self.iradius**2)*(self.C12a*np.cos(2*self.itheta) 
                                                         + self.C12b*np.sin(2*self.itheta) ))
        return CTF


    # Simulates 2D CTF, with alternative representation of twofold astigmatism.
    def __simulate_2D_CTF_stigmated( self ):
        CTF = np.sin( (np.pi*self.defocus*self.lamb*(self.iradius**2) ) + \
                      (0.5*np.pi*self.Cs*(self.lamb**3)*(self.iradius**4)) + \
                      np.pi*self.lamb*(self.iradius**2)* 0.5*((self.C12a + self.C12b) + \
                      (self.C12a - self.C12b)*np.cos(2*(self.itheta - self.phi))) )
        return CTF
    

    # Model of temporal coherence envelope.
    def __temporal_coherence( self ):
        delta = self.Cc * ( self.focal_spread / ( self.kV * 1000 ))
        Et2d = np.exp( -0.25*(( np.pi* self.lamb * delta)**2) * ( self.iradius**4 ) )
        return Et2d
    

    # Model of spatial coherence envelope.
    def __spatial_coherence( self ):
        # dChi is the derivative of the CTF.
        dChi_2d = (2*np.pi*self.lamb*self.iradius*self.defocus) 
        + (2*np.pi*self.Cs*(self.lamb**3)*(self.iradius**3))
        Es2d = np.exp( -(self.beta / ((4*self.lamb**2))) * abs(dChi_2d)**2 )
        return Es2d
    

    # Model of aperture function.
    def __aperture_function( self ):
        aperture_2d = np.ones(( self.imageX, self.imageY ))
        n = range(0, len(aperture_2d[0] ))
        m = range(0, len(aperture_2d[1] ))
        for i in n:
            for j in m:
                if self.iradius[i,j] < self.cutoff:
                    aperture_2d[i,j] = 1
                if self.iradius[i,j] >= self.cutoff:
                    aperture_2d[i,j] = 0
        return aperture_2d
        

    def plot_ctf( self ):
        '''
        Plot the 2D CTF.

        Returns
        -------
        fig
            Matplotlib figure.
        ax
            Matplotlib axis.
        '''
        fig, ax = plt.subplots(1,1)
        ax.matshow( self.square_CTF, cmap='grey')   
        ax.set_xticks([])
        ax.set_yticks([]) 
        return
    

    def update( self ):
        '''
        Update CTF simulation.

        Notes
        -----
        Call after changing class attributes to re-simulate CTF.
        '''
        self.CTF = self.__simulate_2D_CTF_stigmated()
        self.aperture = self.__aperture_function()
        self.temporal = self.__temporal_coherence()
        self.spatial = self.__spatial_coherence()
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2
        return


    def print_aberrations( self ):
        '''
        Print lens aberrations in class.
        '''
        string = ('defocus (nm): ' + str( self.defocus * 1e9 )  + '\n' +
        'C12 (nm, deg): ' + str( self.C12a * 1e9 ) + ", " + str( self.phi )  + '\n' +
        'Cs (mm): ' + str( self.Cs * 1e3 )  + '\n' +
        'Cc (mm):  ' + str( self.Cc * 1e3 )  + '\n')
        print( string )
        return


    def show_all( self ):
        '''
        Plot all simulated arrays.

        Notes
        -----
        Uses matplotlib to plot a figure containing the CTF, damped CTF,
        square CTF, aperture function, temporal coherence function, and
        spatial coherence function. Uses plt.subplots with ax.matshow.
        '''
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].matshow( self.CTF )
        axs[0, 1].matshow( self.damped_CTF )
        axs[0, 2].matshow( self.square_CTF )
        axs[1, 0].matshow( self.aperture )
        axs[1, 1].matshow( self.temporal )
        axs[1, 2].matshow( self.spatial )

        a = [axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
        titles = ['CTF', 'Damped CTF', 'Square CTF', 'Aperture function',\
        'Temporal coherence','Spatial coherence']

        n = 0
        for ax in a:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title( titles[n] )
            n=n+1
        return


class CTFSimulation1D:
    '''
    Class for simulating 1D CTFs.

    Attributes
    ----------
    kV : float
    lamb : float
    flim : float
    fno : float
    cutoff : float
    defocus : float
    frequency : array
    CTF : array
    aperture : array
    temporal : array
    spatial : array
    damped_CTF : array
    square_CTF : array
    focal_spread : float
    Cs : float
    Cc : float
    C12a : float
    C12b : float
    phi : float
    beta : float

    Methods
    ------
    print_aberrations()
    update()
    plot_ctf()
    show_all()

    Notes
    -----
    Simulates 1D CTF. Twofold astigmatism causes peak broadening. 
    '''
    def __init__( self, flim, fno, kV, defocus ):
        '''
        Parameters
        ----------
        flim : array
        fno : int
        kV : float
        defocus : float
        '''
        self.kV = kV
        self.lamb = kv_to_lamb( kV )
        # add lens_aberration class
        self.Cs = 1.6 * 1e-3
        self.Cc = 1.6 * 1e-3
        self.C12a = 0 * 1e-9
        self.C12b = 0 * 1e-9
        self.focal_spread = 5.25
        self.phi = 0 
        self.beta =  0 * 1e-3
        self.flim = flim * 1e9
        self.fno = fno
        self.cutoff = flim * 1e9
        self.defocus = defocus * 1e-9
        self.frequency = self.__calculateFrequencyRange()
        self.CTF = self.__calculate_CTF()
        self.aperture = self.__aperture_function( )
        self.temporal = self.__temporal_coherence( )
        self.spatial= self.__spatial_coherence( )
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2

    def print_aberrations( self ):
        '''
        Print lens aberrations in class.
        '''
        string = ('defocus (nm): ' + str( self.defocus * 1e9 )  + '\n' +
        'C12 (nm, deg): ' + str( self.C12a * 1e9 ) + ", " + str( self.phi )  + '\n' +
        'Cs (mm): ' + str( self.Cs * 1e3 )  + '\n' +
        'Cc (mm):  ' + str( self.Cc * 1e3 )  + '\n')
        print( string )
        return
        

    def update( self ):
        '''
        Update 1D CTF simulation.

        Notes
        -----
        Call method after changing class attributes to update simulation.
        '''
        self.frequency = self.__calculateFrequencyRange()
        self.CTF = self.__calculate_CTF()
        self.aperture = self.__aperture_function( )
        self.temporal = self.__temporal_coherence( )
        self.spatial= self.__spatial_coherence( )
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2
        return


    # Calculates the frequency array for CTF simulation.
    def __calculateFrequencyRange( self ):
        frequency = [0 for _ in range(self.fno)]
        n = range(0, self.fno)
        for i in n:
            frequency[i] = float( (self.flim) * i/self.fno )
        return frequency
    

    # Models the 1D CTF.
    def __calculate_CTF( self ):
        CTF = [0 for _ in range( self.fno )]
        n = range(0, self.fno)
        for i in n:
            f = self.frequency[i]
            CTF[i] = np.sin((np.pi*self.defocus*self.lamb*(f**2)) 
                            +(0.5*np.pi*self.Cs*(self.lamb**3)*(f**4)) 
                            +np.pi*self.lamb*(f**2)*(self.C12a*np.cos(2*self.phi) + self.C12b*np.sin(2*self.phi) ) )
            #frequency[i] = f
        return CTF


    # Models the aperture function.
    def __aperture_function( self ):
        '''
        Calculate 1D aperture function.
        '''
        aperture = np.zeros(len(self.CTF))
        n = range(0, len( aperture ))
        for i in n:
            if self.frequency[i] < self.cutoff:
                aperture[i] = 1
            if self.frequency[i] >= self.cutoff:
                aperture[i] = 0
        return aperture


    # Models the temporal coherence envelope.
    def __temporal_coherence( self ):
        V = self.kV * 1000
        #delta = Cc * np.sqrt( 4* ((deltaI/I)**2) * ((deltaE/V)**2) * ((deltaV/V)**2) )# spatial units
        delta = self.Cc * ( self.focal_spread / V)
        Et = np.zeros(len(self.CTF))
        n = range(0, len( Et ))
        for i in n:
            f = float( self.flim*( i / self.fno ))
            Et[i] = np.exp( -0.25*(( np.pi* self.lamb * delta)**2) * f**4)
        return Et


    # Models the spatial coherence envelope.
    def __spatial_coherence( self ):
        Es = np.zeros(len( self.CTF ))
        dChi = np.zeros(len( self.CTF ))
        n = range(0, len( Es ))
        for i in n:
            f = float( self.flim *( i / self.fno ))
            dChi[i] = (2*np.pi* self.lamb * f * self.defocus)\
            + (2*np.pi*self.Cs*(self.lamb**3)*(f**3))
            Es[i] = np.exp( -(self.beta / ((4*(self.lamb**2)))) * abs(dChi[i])**2 )
        return Es


    def plot_ctf( self ):
        '''
        Plot 1D CTF simulation.

        Returns
        -------
        fig
            Matplotlib figure.
        ax
            Matplotlib axis.

        Notes
        -----
        All simulated values are plotted on same axis, allowing comparison. 
        '''
        ## Plot 1D CTF ##
        fig, ax  = plt.subplots(1,1)
        fig.figaspect=[1,2]
        ax.plot(self.frequency, self.square_CTF, label='CTF$^2$', color='darkviolet')
        # plot aperture function
        ax.plot(self.frequency, self.aperture, label='Aperture', color='orange')
        # plot temporal envelope
        ax.plot(self.frequency, self.temporal, label='Temporal envelope', color='forestgreen')
        # plot spatial envelope
        ax.plot(self.frequency, self.spatial, label='Spatial envelope', color='firebrick')
        # axis settings
        ax.axhline(0, color='black', linewidth=0.5)
        # plot settings
        #ax.set_xlim(0, self.flim*1e9)
        ax.set_yticks([0, 1])
        ax.legend()
        ax.set_ylabel('Intensity', fontsize = 16)
        ax.set_xlabel('Frequency / m-1', fontsize = 16)
        ax.set_box_aspect(1)
        fig.tight_layout()
        return


    def show_all( self ):
        '''
        Plot all simulated arrays.

        Notes
        -----
        Uses matplotlib to plot a figure containing the CTF, damped CTF, square CTF, 
        aperture function, temporal coherence function, and spatial coherence function. 
        Uses plt.subplots with ax.plot().
        '''
        fig, axs = plt.subplots(2, 3, figsize=(8, 8))
        axs[0, 0].plot( self.frequency, self.CTF )
        axs[0, 1].plot( self.frequency, self.damped_CTF )
        axs[0, 2].plot( self.frequency, self.square_CTF )
        axs[1, 0].plot( self.frequency, self.aperture )
        axs[1, 1].plot( self.frequency, self.temporal )
        axs[1, 2].plot( self.frequency, self.spatial )
        a = [axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
        titles = ['CTF', 'Damped CTF', 'Square CTF', 'Aperture function',\
        'Temporal coherence','Spatial coherence']
        n = 0
        for ax in a:
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.set_title( titles[n] )
            ax.set_box_aspect( 1 )
            n=n+1
        #fig.layout=tight
        return