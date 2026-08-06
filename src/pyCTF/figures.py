'''
A class for plotting figures.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#WIP
class zernike_polynomials:
    '''
    Work in progress.
    '''
    def __init__(self):
        return

    def Piston( self, a0, iradius ):
        # Piston - average intensity in the image.
        piston = np.ones([iradius.shape[0], iradius.shape[1]])*a0
        return piston

    def C01( self, a0, iradius, itheta, radius ):
        # Tilt.
        Tx = a0 * (iradius/radius)*np.cos(itheta)
        #Ty = a0 * (iradius/radius)*np.sin(itheta)
        return Tx#, Ty

    def C12( self, a0, iradius, itheta, radius ):
        # Twofold astigmatism.
        C12x = (a0 * (((iradius/radius)**2) * np.sin(2 * itheta )))
        #C12y = (a0 * (((iradius/radius)**2) * np.cos(2 * itheta )))
        return C12x#, C12y

    def defocus( self, a0, iradius, radius ):
        # Defocus.
        defocus = a0 * (2*(iradius/radius)**2 - 1)
        return defocus

    def C21( self, a0, iradius, itheta, radius ):
        # Coma.
        C21x = a0* (3*(iradius/radius)**3 - 2*(iradius/radius))*np.cos(itheta)
        #C21y = a0* (3*(iradius/radius)**3 - 2*(iradius/radius))*np.sin(itheta)
        return C21x#, C21y

    def C23( self, a0, iradius, itheta, radius ):
        # Trself, efoil.
        C23x = a0 * ((iradius/radius)**3)*np.cos(3*itheta)
        #C23y = a0 * ((iradius/radius)**3)*np.sin(3*itheta)
        return C23x#, C23y

    def Cs( self, a0, iradius, radius ):
        # Cs.
        Cs = a0 *( 6*(iradius/radius)**4 - 6*(iradius/radius)**2 + 1 )
        return Cs

    def C34( self, a0, iradius, itheta, radius ):
        # Quadrafoil.
        C32x = a0 *(iradius/radius)**4 *np.cos(4*itheta)  
        #C32y = a0 *(iradius/radius)**4 *np.sin(4*itheta)
        return C32x#, C32y

    def C32( self, a0, iradius, itheta, radius ):
        # Fourfold astigmatism.
        C34x = a0* (4*(iradius/radius)**4 - 3*(iradius/radius)**2)*np.cos(2*itheta)
        #C32y = a0* (4*(iradius/radius)**4 - 3*(iradius/radius)**2)*np.sin(2*itheta)
        return C34x#, C34y

    def _aperture( self, radius, iradius, itheta ):
        aperture = np.ones((iradius.shape[0], iradius.shape[1]) )
        n = range(0, aperture.shape[0] )
        m = range(0, aperture.shape[1] )
        for i in n:
            for j in m:
                if iradius[i,j] < radius:
                    aperture[i,j] = 1
                if iradius[i,j] >= radius:
                    aperture[i,j] = 0
        return aperture


    def _phase_plate( self, radius, iradius, itheta ):
        # Create a phase map.
        a = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        aperture = self._aperture( radius, iradius, itheta )
        polynomials = np.array([self.Piston(a[0], iradius),
                                self.C01(a[1], iradius, itheta, radius),
                                self.defocus(a[3], iradius, radius),
                                self.C12(a[2], iradius, itheta, radius),
                                self.C21(a[4], iradius, itheta, radius),
                                self.C23(a[5], iradius, itheta, radius),
                                self.Cs(a[6], iradius, radius),
                                self.C32(a[7], iradius, itheta, radius),
                                self.C34(a[8], iradius, itheta, radius)])
        polynomials = polynomials * aperture
        return polynomials


    def _plot_phase_map( self, polynomials, scale=1 ):
        # Plot a phase map.
        plate = plate = np.sum(polynomials, 0)
        fig, ax = plt.subplots()
        radius=plate.shape[0]*0.5

        import matplotlib.patches as patches
        circ = patches.Circle((plate.shape[0]*0.5,
                                plate.shape[0]*0.5),
                                radius=radius,
                                facecolor='none')
        ax.add_patch(circ)
        im = ax.imshow(plate, clip_path=circ, clip_on=True, cmap='twilight')
        ax.set_xlabel('Frequency / nm$^{-1}$')
        ax.set_ylabel('Frequency / nm$^{-1}$')
        self._set_ticks( ax, plate, scale )
        return fig, ax


    def _plot_all( self, polynomials ):
        # Plot all polynomials.
        w = 5
        h = 5
        cmap='twilight'

        titles = ['C$_{00}$',
                  'Tilt',
                  'Defocus',
                  'C$_{12}$',
                  'C$_{21}$',
                  'C$_{23}$',
                  'C$_{s}$',
                  'C$_{32}$',
                  'C$_{34}$',
                  'C$_{32}$',
                  'C$_{34}$']

        ax0 = plt.subplot2grid((w,h), (0, 0)) # piston
        ax1 = plt.subplot2grid((w, h), (1, 1)) # tilt
        ax2 = plt.subplot2grid((w, h), (2, 0)) # defocus
        ax3 = plt.subplot2grid((w, h), (2, 2)) # astig
        ax4 = plt.subplot2grid((w, h), (3, 1))
        ax5 = plt.subplot2grid((w, h), (3, 3))
        ax6 = plt.subplot2grid((w, h), (4, 0))
        ax7 = plt.subplot2grid((w, h), (4, 2))
        ax8 = plt.subplot2grid((w, h), (4, 4))

        fig = plt.gcf()
        axs = fig.get_axes()

        import matplotlib.patches as patches
        n = 0
        for ax, poly, title in zip(axs, polynomials, titles):
            circ = patches.Circle((polynomials[0].shape[0]*0.5,
                                polynomials[0].shape[0]*0.5),
                                radius=polynomials[0].shape[0]*0.5,
                                facecolor='none')
            ax.add_patch(circ)
            ax.imshow( poly, clip_path=circ, clip_on=True, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1)
            ax.set_axis_off()
            ax.set_title(title)
        fig.subplots_adjust(wspace=0, hspace=0)
        return fig, axs


    def _set_ticks( self, ax, data, scale=1 ):
        # Set ticks for phase plot.
        ticks = np.array([0,\
            data.shape[0]*0.25,\
            data.shape[0]*0.5,\
            data.shape[0]*0.75,\
            data.shape[0]])

        labels = np.array([ str(-data.shape[0]*scale),\
                            str(-data.shape[0]*0.5*scale),\
                            str(0),\
                            str(data.shape[0]*0.5*scale),\
                            str(data.shape[0]*scale) ])

        ax.set_xticks( ticks, labels=labels)
        ax.set_yticks( ticks, labels=labels)
        return


def plot_profiles( ElectronImage ):
    '''
    Show the results of measuring radial profiles.
    '''
    fig, axs = plt.subplots( 1, 2, figsize=(8,8) )
    axs[0].plot( ElectronImage.frequency, ElectronImage.radial_profile, label='Radial profile' )
    axs[0].plot( ElectronImage.cropped_frequency, ElectronImage.baseline, label='Baseline' )
    axs[0].plot( ElectronImage.cropped_frequency, ElectronImage.cropped_profile, label='Cropped' )
    axs[1].plot( ElectronImage.cropped_frequency, ElectronImage.smoothed_profile, label='Smoothed profile')
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)
    axs[0].set_ylabel('Intensity / a.u.', fontsize = 16)
    axs[0].set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
    axs[1].set_xlabel('Frequency / $nm^{-1}$', fontsize = 16)
    axs[0].legend()
    axs[1].legend()
    return


def plot_background( ElectronImage, cmap='cividis' ):
    '''
    Plot results of Fourier background removal.

    Notes
    -----
    Creates a plot showing the processed CTF, the low frequency
    background, and the envelope background.
    '''
    fig, axs = plt.subplots(1, 3, figsize=(8,8))
    axs[0].matshow( ElectronImage.image, cmap=cmap )
    axs[1].matshow( ElectronImage.LF_bkg, cmap=cmap )
    axs[2].matshow( ElectronImage.E_bkg, cmap=cmap )

    axs[0].set_title( 'Background removed' )
    axs[1].set_title( 'Low frequency' )
    axs[2].set_title( 'Envelope function' )

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    try:
        scalebar = make_scalebar( 0.5, ElectronImage.scale, axs[0] )
        axs[0].add_artist(scalebar)
        scalebar = make_scalebar( 0.5, ElectronImage.scale, axs[1] )
        axs[1].add_artist(scalebar)
        scalebar = make_scalebar( 0.5, ElectronImage.scale, axs[2] )
        axs[2].add_artist(scalebar)
    except:
        print('Error: could not add scalebar to image.')
    return