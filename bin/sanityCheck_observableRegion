#!/usr/bin/python
description = "makes sanity check plots for observableRegion algorithm"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import simUtils as utils
import numpy as np
import healpy as hp

import matplotlib
import pylab as plt

#-------------------------------------------------

nside = 128

t = 1149984017.

obsAng = np.pi/3

obsLat = np.pi/3
obsLon = None

occAng = np.pi/6

#-------------------------------------------------

obsReg = utils.observableRegion( t, nside, obsAng, obsLat, obsLon, solar_occlusion_angle=occAng )
mask = utils.solarOcclusion( t, nside, occAng )
masK = utils.solarOcclusion( t, nside, occAng+obsAng )

#-------------------------------------------------

hp.mollview( mask, title='solar occlusion marginalized over latitude' )
hp.graticule()
plt.show()

hp.mollview( masK, title='location of the effective terminator' )
hp.graticule()
plt.show()

hp.mollview( obsReg, title='observable region' )
hp.graticule()
plt.show()

hp.mollview( mask + 2*masK + 3*obsReg, title='overlap' )
hp.graticule()
plt.show()
