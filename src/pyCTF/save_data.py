'''
Module containing containing functions to save data to file.

Work in progress.
'''

import numpy as np


def save_CTF_profile_to_file( CTF, **kwargs ):
    extension = kwargs.get('extension', '.msa')
    output = _create_msa_from_array( CTF.smoothed_profile, 
        CTF.cropped_frequency, 
        scale=CTF.scale )
    _save_to_file( output, extension, 'file' )
    return


def _save_to_file( data, extension, filename ):
    name = filename + extension
    with open(name, "x", encoding="utf-8") as f:
        f.write(data)
    return


# Function to create csv file from xy file.
def _xy_to_csv( xy ):
    rows = ["{}, {}".format(i, j) for i, j in xy]
    text = "\n".join(rows)
    return text

def _create_xy_from_array( x, y ):
    output = np.stack( (x, y), axis=0)
    output = np.transpose( output )
    output = np.round(output, 3)
    return output


# Functions for msa files.
def _create_msa_header( npoints,
                        xunits,
                        xperchannel,
                        offset ):
    header = ''\
    '#FORMAT         : EMSA/MAS Spectral Data File'+'\n'\
    '#VERSION     : 1.0'+'\n'\
    '#TITLE       : line projection'+'\n'\
    '#DATE        : '+'\n'\
    '#TIME        : '+'\n'\
    '#OWNER       : '+'\n'\
    '#NPOINTS     : '+ npoints +'\n'\
    '#NCOLUMNS    : 1'+'\n'\
    '#XUNITS      : '+ xunits +'\n'\
    '#YUNITS      : '+'\n'\
    '#DATATYPE    : XY'+'\n'\
    '#XPERCHAN    : '+ xperchannel +'\n'\
    '#OFFSET      : '+ offset +'\n'\
    '#SPECTRUM    : Spectral Data Starts Here\n'
    return header


def _create_msa_footer():
    footer = '\n#ENDOFDATA   : End Of Data and File'
    return footer


def _assemble_msa( header, footer, xy ):
    output = header + xy + footer
    return output


def _create_msa_from_array( xdata, ydata, **kwargs ):
    xunits = kwargs.get('xunits', '1/nm')
    scale = kwargs.get('scale', 1.0)
    npoints = str(np.size( xdata ))
    xperchannel = str( np.round( abs(xdata[1]-xdata[0]), 3) )
    header = _create_msa_header( npoints, xunits, xperchannel )
    footer = _create_msa_footer()
    xy = _create_xy_from_array( xdata, ydata )
    xy = _xy_to_csv( xy )
    output = _assemble_csv( header, footer, xy )
    return output