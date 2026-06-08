'''
Classes for simulating contrast transfer functions.

This module contains classes for simulating 2D and 1D CTFs.
'''

import numpy as np 
import matplotlib.pyplot as plt

from pyCTF.utils import kv_to_lamb
from pyCTF.utils import LensAberrations
from pyCTF.utils import LineProfiles
from pyCTF.utils import normalise_data_range


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
    def __init__( self, flim, image_size, kV, defocus, **kwargs ):
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
        LineProfiles.__init__( self )
        self.cutoff = kwargs.get('aperture', flim) *1e9
        self.temporal_mode = kwargs.get('mode', 0)
        self.Cs = kwargs.get('Cs', 1.6) * 1e-3
        self.Cc = kwargs.get('Cc', 1.6) * 1e-3
        self.C12a = kwargs.get('C12a', 0) * 1e-9
        self.C12b = kwargs.get('C12b', 0) * 1e-9
        self.phi = kwargs.get('phi', 0) 
        self.beta =  kwargs.get('beta', 0) * 1e-3
        # Objective current instability.
        self.dI = kwargs.get('delta_current',0)
        self.I = kwargs.get('current', 0)
        # Source voltage instability, in Volts.
        self.dV = kwargs.get('delta_voltage', 0)
        # Electron energy spread in electron-volts.
        self.dE = kwargs.get('delta_E',0)
        self.kV = kV
        self.lamb = kv_to_lamb( kV )
        self.E = self.kV*1000
        # Focal spread.
        self.focal_spread = kwargs.get('focal_spread', 5.25)
        ## params ##
        self.image_size = image_size
        self.flim = flim * 1e9
        self.defocus = defocus * 1e-9
        self.scale = self.image_size / self.flim
        ## images ##
        self.imageX = image_size
        self.imageY = image_size
        self.CTF = np.ones(( self.imageX, self.imageY ))
        self.cent = np.array([ self.CTF.shape[0] / 2.0, self.CTF.shape[0] ]) 
        self.iradius, self.itheta = self.__find_radial_distance( )
        ## simulations ##
        self.update()


    def update( self ):
        '''
        Update CTF simulation.

        Notes
        -----
        Call after changing class attributes to re-simulate CTF.
        '''
        self.CTF = self.__simulate_2D_CTF_stigmated()
        self.aperture = self.__aperture_function()
        self.temporal = self.__temporal_coherence( self.temporal_mode )
        self.spatial = self.__spatial_coherence()
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2
        self.square_CTF = normalise_data_range( self.square_CTF )
        return
    

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
    def __simulate_2D_CTF_stigmated( self ):
        CTF = np.sin( (np.pi*self.defocus*self.lamb*(self.iradius**2) )+\
                      (0.5*np.pi*self.Cs*(self.lamb**3)*(self.iradius**4))+\
                      np.pi*self.lamb*(self.iradius**2)*0.5*((self.C12a + self.C12b)+\
                      (self.C12a - self.C12b)*np.cos(2*(self.itheta - self.phi))) )
        return CTF
    

    # Calculate focal spread. W&C p.471.
    def __calculate_focal_spread( self, mode ):
        try:
            if mode == 1:
                # Objective current instability.
                Iq = self.dI / self.I
                # Source voltage instability.
                Iv = self.dV / (self.kV*1000)
                # Electron energy spread.
                Ie = self.dE / self.E
                delta = self.Cc * np.sqrt( 4 * (Iq**2) + (Iv**2) + (Ie**2) )
            if mode == 0:
                delta = self.Cc*(self.focal_spread/(self.kV*1000))
        except:
            print('Error: could not caculate focal spread.')
        return delta


    # Model of temporal coherence envelope.
    def __temporal_coherence( self, mode ):
        delta = self.__calculate_focal_spread( mode )
        delta = self.Cc * ( self.focal_spread / ( self.kV * 1000 ))
        Et2d = np.exp( -0.25*(( np.pi* self.lamb * delta)**2) * ( self.iradius**4 ) )
        return Et2d
    

    # Model of spatial coherence envelope.
    def __spatial_coherence( self ):
        preexp = ((np.pi*self.beta)/self.lamb)**2
        dChi_2d = ((self.Cs*(self.lamb**3)*(self.iradius**3)\
            +(self.defocus*self.lamb*self.iradius) ))**2
        Es2d = np.exp( preexp * dChi_2d )
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
        cax = ax.matshow( self.square_CTF, cmap='grey')   
        ax.set_xticks([])
        ax.set_yticks([]) 
        # Note: fix max extent of colorbar.
        cbar = fig.colorbar( mappable=cax )#, ticks=[0,0.5,0.99] )
        #cbar.ax.set_yticklabels(['0.0', '0.5', '0.99'])
        return
    

    def print_aberrations( self ):
        '''
        Print lens aberrations in class.
        '''
        string = ('defocus (nm): ' + str( self.defocus * 1e9 )+'\n'+
        'C12 (nm, deg): ' + str( self.C12a * 1e9 )+", "+\
        str( self.C12b * 1e9 )+\
        ", "+str( self.phi )+'\n'+
        'Cs (mm): ' + str( self.Cs * 1e3 )  + '\n' +
        'Cc (mm):  ' + str( self.Cc * 1e3 )  + '\n')
        print( string )
        return


    def show_all( self, **kwargs ):
        '''
        Plot all simulated arrays.

        Notes
        -----
        Uses matplotlib to plot a figure containing the CTF, damped CTF,
        square CTF, aperture function, temporal coherence function, and
        spatial coherence function. Uses plt.subplots with ax.matshow.
        '''
        fig, axs = plt.subplots(2, 3)
        axs[0, 0].matshow( self.CTF, cmap='grey' )
        axs[0, 1].matshow( self.damped_CTF, cmap='grey' )
        axs[0, 2].matshow( self.square_CTF, cmap='grey' )
        axs[1, 0].matshow( self.aperture, cmap='grey' )
        axs[1, 1].matshow( self.temporal, cmap='grey' )
        axs[1, 2].matshow( self.spatial, cmap='grey' )

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
    def __init__( self, flim, fno, kV, defocus, **kwargs ):
        '''
        Parameters
        ----------
        flim : array
        fno : int
        kV : float
        defocus : float
        mode : int
            Flag for how to calculate temporal coherence.
        '''
        self.cutoff = kwargs.get('aperture', flim) *1e9
        self.kV = kV
        self.lamb = kv_to_lamb( kV )
        # add lens_aberration class
        self.Cs = kwargs.get('Cs', 1.6) * 1e-3
        self.Cc = kwargs.get('Cc', 1.6) * 1e-3
        self.C12a = kwargs.get('C12a', 0) * 1e-9
        self.C12b = kwargs.get('C12b', 0) * 1e-9
        # Objective current instability.
        self.dI = kwargs.get('delta_current',0)
        self.I = kwargs.get('current', 0)
        # Source voltage instability, in Volts.
        self.dV = kwargs.get('delta_voltage', 0)
        # Electron energy spread in electron-volts.
        self.dE = kwargs.get('delta_E',0)
        self.E = self.kV*1000
        # Focal spread.
        self.focal_spread = kwargs.get('focal_spread', 5.25)
        self.phi = kwargs.get('phi', 0) 
        self.beta =  kwargs.get('beta', 0) * 1e-3
        self.flim = flim * 1e9
        self.fno = fno
        #self.cutoff = flim * 1e9
        self.defocus = defocus * 1e-9
        self.frequency = self.__calculateFrequencyRange()
        self.CTF = self.__calculate_CTF()
        self.aperture = self.__aperture_function( )
        self.temporal_mode = kwargs.get('mode', 0)
        self.temporal = self.__temporal_coherence( self.temporal_mode )
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
        self.aperture = self.__aperture_function()
        self.temporal = self.__temporal_coherence( self.temporal_mode )
        self.spatial= self.__spatial_coherence()
        self.damped_CTF = self.CTF * self.temporal * self.spatial * self.aperture
        self.square_CTF = self.damped_CTF**2
        return


    # Calculates the frequency array for CTF simulation.
    def __calculateFrequencyRange( self ):
        frequency = [0 for _ in range(self.fno)]
        n = range(0, self.fno)
        for i in n:
            frequency[i] = float( (self.flim) * i/self.fno )
            frequency = np.array(frequency)
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


    # Calculate focal spread. W&C p.471.
    def __calculate_focal_spread( self, mode ):
        try:
            if mode == 1:
                # Objective current instability.
                Iq = self.dI / self.I
                # Source voltage instability.
                Iv = self.dV / (self.kV*1000)
                # Electron energy spread.
                Ie = self.dE / self.E
                delta = self.Cc * np.sqrt( 4 * (Iq**2) + (Iv**2) + (Ie**2) )
            if mode == 0:
                delta = self.Cc*(self.focal_spread/(self.kV*1000))
        except:
            print('Error: error calculating focal spread.')
            delta = 0
        return delta


    # Models the temporal coherence envelope.
    # Update to use physical quants.
    def __temporal_coherence( self, mode ):
        V = self.kV * 1000
        Et = np.zeros(len(self.CTF))
        n = range(0, len( Et ))
        delta = self.__calculate_focal_spread( mode )
        for i in n:
            f = float( self.flim*( i / self.fno ))
            Et[i] = np.exp( -0.25*(( np.pi* self.lamb * delta)**2) * f**4)
        return Et


    # Models the spatial coherence envelope.
    # W&C p. 471
    def __spatial_coherence( self ):
        Es = np.zeros(len( self.CTF ))
        dChi = np.zeros(len( self.CTF ))
        n = range(0, len( Es ))
        preexp = ((np.pi*self.beta)/self.lamb)**2
        for i in n:
            f = float( self.flim *( i / self.fno ))
            dChi[i] = ((self.Cs*(self.lamb**3)*(f**3) +\
                (self.defocus*self.lamb*f) ))**2
            Es[i] = np.exp( preexp * -dChi[i] )
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
        #freq = [x*1e-9 for x in self.frequency]
        ax.plot((self.frequency)*1e-9,
            self.square_CTF,
            label='CTF$^2$',
            color='darkviolet')
        # plot aperture function
        ax.plot(self.frequency*1e-9,
            self.aperture,
            label='Aperture',
            color='orange')
        # plot temporal envelope
        ax.plot(self.frequency*1e-9,
            self.temporal,
            label='Temporal envelope',
            color='forestgreen')
        # plot spatial envelope
        ax.plot(self.frequency*1e-9,
            self.spatial,
            label='Spatial envelope',
            color='firebrick')
        # axis settings
        ax.axhline(0, color='black', linewidth=0.5)
        # plot settings
        #ax.set_xlim(0, self.flim*1e9)
        ax.set_yticks([0, 1])
        ax.legend()
        ax.set_ylabel('Intensity', fontsize = 16)
        ax.set_xlabel('Frequency / nm$^{-1}$', fontsize = 16)
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
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        axs[0, 0].plot( self.frequency*1e-9, self.CTF, color='k' )
        #axs[0, 0].set_ylim([-1,1])
        axs[0, 1].plot( self.frequency*1e-9, self.damped_CTF, color='k' )
        #axs[0, 1].set_ylim([-1,1])
        axs[0, 2].plot( self.frequency*1e-9, self.square_CTF, color='k' )
        #axs[0, 2].set_ylim([0,1])
        axs[1, 0].plot( self.frequency*1e-9, self.aperture, color='k' )
        #axs[1, 0].set_ylim([0,1.01])
        axs[1, 1].plot( self.frequency*1e-9, self.temporal, color='k' )
        #axs[1, 1].set_ylim([0,1.01])
        axs[1, 2].plot( self.frequency*1e-9, self.spatial, color='k' )
        axs[1, 2].set_ylim([0,1.01])
        a = [axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]
        titles = ['CTF', 'Damped CTF', 'Square CTF', 'Aperture function',\
        'Temporal coherence','Spatial coherence']
        n = 0
        for ax in a:
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.set_title( titles[n] )
            ax.set_box_aspect( 1 )
            ax.set_xlabel('Frequency / nm$^{-1}$')
            ax.set_ylabel('Intensity', fontsize=10)
            n=n+1
        fig.tight_layout()
        return