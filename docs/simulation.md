# Simulating contrast transfer functions

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
