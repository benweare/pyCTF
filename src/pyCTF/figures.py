'''
A class for plotting figures.
'''

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