# pyCTF

pyCTF allows manipulation of experimental contrast transfer functions (CTFs), primarily to measure lens aberrations in transmission electron microscopy (TEM). See the Jupyter Notebook for worked examples.

The following objective lens aberrations can be measured:
- Defocus
- Spherical aberration
- Chromatic aberration

Other functionality:
- Remove background noise
- Simulating CTFs (1D and 2D)

We welcome feedback and suggestions for improvements!

## Installation

pyCTF can be installed with pip.

## Simulating contrast transfer functions

A thorough dicussion of the mathematics of CTFs can be found in the literature (Brydson, 2011). Briefly, to simulate a CTF as seen in experimental electron micrographs, we need to model the following as functions of spatial frequency ($\bf{u}$):

- Squared phase function, $\chi(\bf{u})^2$
- Aperture function, $A(\bf{u})$
- Spatial coherence envelope, $S(\bf{u})$
- Temporal coherence envelope, $E_t(\bf{u})$

Where $AST$ is the envelope damping function, $E(\bf{u})$. The simulated CTF is the product of these functions:

$$
CTF^2 = E(\bf{u})\chi(\bf{u})^2
$$

Currently, simulated CTFs allow modelling of the following geometric lens aberrations: 
- Defocus
- Spherical aberration
- Twofold astigmatism

Other modelled parameters:
- Accelerating voltage
- Focal spread
- Chromatic aberration

## Measuring lens aberrations

### Defocus and spherical aberration

Defocus is measured by extracting and fitting the minima of the CTF (a.k.a. "zeros"). The approach used here is based on the method of Krivanek (Krivanek, 1976) developed by Coene (Coene, 1991), and the description given by Zou (Zou *et al*, 2011).

The spatial frequency of each minima ($u_0$) is extracted, then the following equation is used to plot a straight line: 

$$
\frac{n}{u_0^2}  = \frac{Cs\lambda^2}{2} u + \epsilon
$$

The defocus is given by the y-intercept ($\epsilon$), and gradient can be rearranged for the spherical abberation.

### Twofold astigmatism

pyCTF can currently measure the angle of twofold astigmatism using cross-correlation the CTF with it's mirror image (Rohou and Grigorieff, 2015).

### Chromatic aberration

Chromatic aberration ($C_c$) can be measured using the following relationship (Klemperer, 1971): 

$$
\Delta F = C_c ( \frac{\Delta V}{ V} -  \frac{2\Delta I}{ I }  )
$$

Where $I$ is the objective lens current, $V$ is the accelerating voltage, $\Delta V$ is the voltage instability, and $\Delta F$ is the change in lens focal length. Assuming the objective lens current instability ($\Delta I$) is negligable, measuring the change in defocus as a function of the accelerating voltage gives the chromatic aberration of the objective lens (McMullan *et al*, 2023):

$$
\Delta f = C_c \Delta V
$$

Where $\Delta f$ is the change in defocus.

### Experimental details

Transmission electron microscopy was performed at 200 kV on a JEOL 2100F TEM with a Gatan Model 1027 K3-IS direct detection camera with DigitalMicrograph software (v3.60). CTFs were acquired ultra-thin carbon film on copper mesh TEM grids (EM Resolutions). Images were converted from .dm4 format to .tiff format using ImageJ or DigitalMicrograph.

### Acknowledgements

This work was supported by the EPSRC \[EP/W006413/1 and EP/L022494/1\].

### Citations
- Baek SJ, Park A, Ahn YJ, Choo J. Baseline correction using asymmetrically reweighted penalized least squares smoothing. The Analyst. 2015;140(1):250–7.
- Barthel J, Thust A. Aberration measurement in HRTEM: Implementation and diagnostic use of numerical procedures for the highly precise recognition of diffractogram patterns. Ultramicroscopy. 2010 Dec;111(1):27–46.
- Brydson R, editor. Aberration-corrected analytical transmission electron microscopy. Chichester, West Sussex: RMS-Wiley; 2011.
- Coene WMJ, Denteneer TJJ. Improved methods for the determination of the spherical aberration coefficient in high-resolution electron microscopy from micrographs of an amorphous object. Ultramicroscopy. 1991 Dec;38(3–4):225–33.
- Klemperer OE, Barnett ME. Electron optics. Third ed., first paperback ed. Cambridge: Cambridge Univ. Press; 2010. 506 p. (Cambridge monographs on physics).
- Krivanek OL, Gaskell PH, Howie A. Seeing order in ‘amorphous’ materials. Nature. 1976 Aug;262(5568):454–7.
- McMullan G, Naydenova K, Mihaylov D, Yamashita K, Peet MJ, Wilson H, et al. Structure determination by cryoEM at 100 keV. Proc Natl Acad Sci. 2023 Dec 5;120(49):e2312905120.
- Rohou A, Grigorieff N. CTFFIND4: Fast and accurate defocus estimation from electron micrographs. J Struct Biol. 2015 Nov;192(2):216–21.
- Zou X, Hovmöller S, Oleynikov P. Electron Crystallography: Electron Microscopy and Electron Nanodiffraction. Vol. 1. New York: Oxford University Press; 2012.
