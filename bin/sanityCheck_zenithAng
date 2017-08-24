#!/usr/bin/python
description = "makes sanity check plots for zenithAng algorithm"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import os
import json

import numpy as np
import healpy as hp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    })

from lalinference import plot as lalinf_plot
from lalinference import cmap

from lal.gpstime import tconvert

from astropy.time import Time as astropyTime
from astropy import coordinates

import pyfits

import simUtils as utils

#-------------------------------------------------

verbose = True
seed = 1
np.random.seed(seed)

ndetect = 20

nside = 32
npix = hp.nside2npix( nside )

amp    = 0.4
phase  = np.pi/8.

obsAng = np.pi/2.
solarOcclusionAng = (90+18)*np.pi/180 - obsAng

start = astropyTime('2016-12-01T00:00:00', format='isot', scale='utc').gps
end   = astropyTime('2017-01-01T00:00:00', format='isot', scale='utc').gps

#-------------------------------------------------

### draw an events randomly (NOT uniformly in volume, but good enough for testing!)
srcPix = np.random.randint(low=0, high=npix, size=ndetect)
srcTheta, srcPhi = hp.pix2ang( nside, srcPix )
srcDec   = utils.pi2 - srcTheta

### draw times 
gps = utils.drawTimes( start, end, ndetect, amplitude=amp, phase=phase )

#-------------------------------------------------

### place observatories according to healpix decomposition
pix = np.arange(npix)
obsTheta, obsPhi = hp.pix2ang( nside, pix )
obsLat = utils.pi2 - obsTheta

set_obsLat = sorted(set(obsLat), reverse=True)

#-------------------------------------------------

geojson_filename = os.path.join(os.path.dirname(lalinf_plot.__file__),
    'ne_simplified_coastline.json')
with open(geojson_filename, 'r') as geojson_file:
    geojson = json.load(geojson_file)

### compute delay times
for i in xrange(ndetect):

    print "detection %d"%(i)

    t = float(gps[i])
    this_srcDec = float(srcDec[i])
    this_srcPhi = float(srcPhi[i])
    this_srcTheta = utils.pi2 - this_srcDec
    this_srcRA = utils.rotateRAE2C( this_srcPhi, t )

    # get solar position in spherical coordinate
    timeObj = astropyTime( tconvert(int(t), form="%Y-%m-%dT%H:%M:%S")+("%.3f"%(t%1))[1:], format='isot', scale='utc')
    sun = coordinates.get_sun(timeObj)
    sunDec = sun.dec.radian
    sunTheta = utils.pi2 - sunDec
    sunRA = sun.ra.radian
    sunPhi = utils.rotateRAC2E( sunRA, t )

    filename = 'sanityCheck_zenithAng-%d.txt'%(i)
    print "  "+filename

    file_obj = open(filename, 'w')
    print >> file_obj, "gps : %.3f"%(t)
    print >> file_obj, "srcDec : %.3f"%(this_srcDec)
    print >> file_obj, "srcRA  : %.3f"%(this_srcRA)
    print >> file_obj, "srcPhi : %.3f"%(this_srcPhi)
    print >> file_obj, "%5s %12s"%('lat', 'zenithAng')

    print "    iterating over observatories"
    zenithAng = np.zeros((npix,), dtype=float)
    for lat in set_obsLat:
        if verbose:
            print "      lat=%.1f"%(lat*utils.rad2deg)
        ans = utils.zenithAng( lat, this_srcDec, this_srcRA, sunDec, sunRA, t, obsAng, solarOcclusionAng=solarOcclusionAng ) * utils.rad2deg
        zenithAng[obsLat==lat] = ans
        print >> file_obj, "%3.1f %8.3f"%(lat*utils.rad2deg, ans) 

    file_obj.close()

    #--------------------

    figname = 'sanityCheck_zenithAng-%d.png'%(i)
    print "  "+figname

    fig = plt.figure(figsize=(9,5))
    ax = plt.subplot(111, projection='mollweide')

    ### decorate
    ax.grid(True, which='both')

    for shape in geojson['geometries']:
        verts = np.deg2rad(shape['coordinates'])
        color='k'
        ax.plot(verts[:, 0], verts[:, 1], color=color, linewidth=0.5)

    ### plot zenith Angle
    lalinf_plot.healpix_heatmap( zenithAng, cmap=plt.get_cmap('OrRd'), nest=False, vmin=0., vmax=180 )

    cb = plt.colorbar(orientation='horizontal', pad=0.03, fraction=0.10, shrink=0.85 )
    cb.set_label(r'$D_\mathrm{zenith}$ [$^\circ$]')
    cb.set_ticks([0, 30, 60, 90, 120, 150, 180])

    ### black out positions where observatories cannot observe the true source
    mask = zenithAng==np.infty ### places that can never observe the true source
    mask = np.ma.masked_where(mask!=1, mask)
    lalinf_plot.healpix_heatmap( mask, cmap=plt.get_cmap('Greys'), vmin=0, vmax=1, nest=False, alpha=0.25 )

    ### plot location of the sun
    if sunPhi >= np.pi:
        sunPhi -= utils.twopi
    ax.plot( sunPhi, sunDec, marker='o', markerfacecolor='none', markeredgecolor='k', markersize=10, alpha=1.00 )
    ax.plot( sunPhi, sunDec, marker='.', markerfacecolor='k', markeredgecolor='k', markersize=1, alpha=1.00 )

    ### plot source location
    if this_srcPhi > np.pi:
        this_srcPhi -= utils.twopi
    ax.plot( this_srcPhi, this_srcDec, marker='*', markerfacecolor='k', markeredgecolor='k', markersize=10, alpha=1.00 )

    fig.savefig(figname)

    #--------------------

    figname = 'sanityCheck_zenithAng_SolarOcclusion-%d.png'%(i)
    print "  "+figname

    ### plot solar occlusion at time of detection
    mask = np.cos(solarOcclusionAng) >= np.cos(obsTheta)*np.cos(sunTheta) + np.sin(obsTheta)*np.sin(sunTheta)*np.cos(obsPhi-sunPhi)
    mask = np.ma.masked_where(mask==1, mask)
    lalinf_plot.healpix_heatmap( mask, cmap=plt.get_cmap('Greys_r'), vmin=0, vmax=1, nest=False, alpha=0.10 )

    mask = np.cos(solarOcclusionAng+obsAng) >= np.cos(obsTheta)*np.cos(sunTheta) + np.sin(obsTheta)*np.sin(sunTheta)*np.cos(obsPhi-sunPhi)
    mask = np.ma.masked_where(mask==1, mask)
    lalinf_plot.healpix_heatmap( mask, cmap=plt.get_cmap('Greys_r'), vmin=0, vmax=1, nest=False, alpha=0.10 )

    fig.savefig(figname)

    #--------------------

    plt.close(fig) 
