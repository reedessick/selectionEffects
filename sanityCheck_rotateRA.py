#!/usr/bin/python
description = "makes sanity check plots for rotateRA algorithm"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import json

import sys
import os

import healpy as hp
import numpy as np

import simUtils as utils

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from lalinference import plot as lalinf_plot
from lalinference import cmap

#-------------------------------------------------

algorithm = "LIB"

### load pointers

library = []
for index in sys.argv[1:]:
    rootdir = os.path.dirname( index ) ### used to determine full paths to FITS files
    file_obj = open(index, "r")
    cols = dict( (c, i) for i, c in enumerate(file_obj.readline().strip().split()) ) ### get column headers

    if not cols.has_key(algorithm): ### ensure we have LIB maps
        raise ValueError("could not find algorithm=\"%s\" as a column header in %s"%("LIB", index))
    if not cols.has_key("gps"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))
    if not cols.has_key("RA"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))
    if not cols.has_key("Dec"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))

    for line in file_obj:
        fields = line.strip().split()
        fits = fields[cols[algorithm]]
        if fits!="none": ### there is a map for this injection
            fits = os.path.join(rootdir, fits)
            gps = float(fields[cols['gps']])
            ra = float(fields[cols['RA']])
            dec = float(fields[cols['Dec']])
            print "%d -> %s"%(gps, fits)
            library.append( {'gps':gps, 'fits':fits, 'ra':ra, 'dec':dec} )
    Nm = len(library)

#-------------------------------------------------

### stack posteriors
nside = 512
npix = hp.nside2npix(nside)
sumPost = np.zeros((npix,), dtype=float)

### rotate to EarthFixed
for d in library[:-1]:
    fits = d['fits']
    gps = d['gps']

    print fits

    ### load in, rotate to EarthFixed Frame, and stack
    sumPost += utils.rotateMapC2E( hp.ud_grade( hp.read_map( fits, verbose=False ), nside_out=nside, power=-2 ), gps )

#-------------------------------------------------

fig = plt.figure()
ax = plt.subplot(1,1,1, projection="mollweide")

lalinf_plot.healpix_heatmap( sumPost, cmap=plt.get_cmap("OrRd") )

geojson_filename = os.path.join(os.path.dirname(lalinf_plot.__file__),
    'ne_simplified_coastline.json')
with open(geojson_filename, 'r') as geojson_file:
    geojson = json.load(geojson_file)
for shape in geojson['geometries']:
    verts = np.deg2rad(shape['coordinates'])
    color='k'
    plt.plot(verts[:, 0], verts[:, 1], color=color, linewidth=0.5)

figname = 'sanityCheck_rotateRA.png'
print figname
fig.savefig(figname)
plt.close(fig)
