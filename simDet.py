#!/usr/bin/python
usage = "simDet.py [--options] startGSP endGPS catalogIndex.txt catalogIndex.txt catalogIndex.txt ..."
description = """simulates a set of detections and reports statistics about our localization capabilities

We model the diurnal lock cycle as

  p(t|locked) = (1 + A*sin(2*pi*t/day - phi))/day

This script allows you to simulate detections (from a library of localizations) which will be drawn from 
a distribution in time according to the lock cyle. The script then computes localization statistics 
including the occultation of the Sun and other effects.

Importantly! If you provide multiple catalogIndex.txt files, the script randomly draws from all the files 
together so that the number of events in each file acts as an prior that the type of detection that will 
be simulated. Furthermore, any intrinsic biases in the catalogs will be mirrored into this simulation!

Also, we assume that the filenames stored in the catalogIndex.txt files for each FITS file are relative 
to the directory in which the catalogIndex.txt file lives!

We also assume that the maps stored in the catalog are in Celestial Equatorial coordinates.
"""
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import os
import sys

import pickle

import simUtils as utils

import numpy as np
import healpy as hp

from optparse import OptionParser

#-------------------------------------------------

parser = OptionParser(usage=usage, description=description)

### options about verbosity
parser.add_option("-v", "--verbose", default=False, action="store_true")
parser.add_option("-V", "--Verbose", default=False, action="store_true", help="print very detailed statements")

### parameters of the experiment
parser.add_option("-N", "--Ntrials", dest="Nt", default=1, type="int", help="the number of trials we simulate")
parser.add_option("-n", "--Ndetections", dest="Nd", default=5, type="int", help="the number of detections per trial we simulate")

parser.add_option("", "--seed", default=None, type="int", help="used to set the numpy.random seed")

parser.add_option("", "--nside-exp", default=8, type="int", help="when stacking maps and comparing them, we resample the maps so that they are at nside=2**nside_exp")

### which maps to use
parser.add_option("-a", "--algorithm", default="LIB", type="string", help="the localization algorithm we use for skymaps. This MUST be a column header in the catalogIndex.txt file")

### parameters about the diurnal cycle
parser.add_option("-p", "--phase", default=np.pi/8, type="float", help="the phase (relative to 00:00:00 UTC) in the sinusoidal model of the diurnal lock cycle")
parser.add_option("-A", "--amplitude", default=0.5, type="float", help="the amplitude of the sinusoidal model for the diurnal lock cycle")

### parameters about the EM observatories
parser.add_option("-O", "--observation-angle", default=60, type="float", help="the maximum angle from the zenith which an observatory can observe. Should be specified in degrees.")

parser.add_option("-S", "--observatory", dest="knownObservatory", default=[], action="append", type="string", help="a known observatory. Must be one of : 'Palomar','Cerro Tololo','Mauna Kea','La Palma','Anglo-Australian','Nishi-Harima'")

parser.add_option("-s", "--observatory-latitude-longitude", dest="observatory", nargs=3, default=[], action="append", type="string", help="the observatory name longitude and latitue as space separated strings, eg: Palomar XXX YYY. Latitude and Longitude should be specified in degrees.")

### parameters about how much the sun affects observatories
parser.add_option("-o", "--solar-occlusion-angle", default=30, type="float", help="the closest angle to the sun that is still observable. Should be specified in degrees so that the Sun will block out half the sky when --occlusion-angle=90.")
parser.add_option("-m", "--lunar-occlusion-angle", default=10, type="float", help="the closest angle to the sun that is still observable. Should be specified in degrees so that the Sun will block out half the sky when --occlusion-angle=90.")

### parameters about saving the result
parser.add_option("-d", "--output-dir", default=".", type="string")
parser.add_option("-t", "--tag", default="", type="string")

parser.add_option("-e", "--expensive-logging", default=False, action="store_true", help="saves the posterior and the skymap in addition to the statistics derived therefrom. Also stores exactly one copy of the observatories' observableRegions as a second dump into the pickle file.")
parser.add_option("", "--memory-light", default=False, action="store_true", help="forget about posteriors after each event for each trial. Increases I/O burden but keeps the memory light")

### parameters to speed things up
parser.add_option("", "--timeUntilUnOc", default=False, action="store_true", help="if not supplied, will skip this step (which appears to be the most computationally expensive).")

opts, args = parser.parse_args()

### finish parsing input arguments
if len(args) < 3:
    raise ValueError("please supply at least 3 input arguments\n%s\n%s"%(usage,description))

start = float(args[0])
end = float(args[1])

catalogs = args[2:]

### verbosity statements
opts.verbose = opts.verbose or opts.Verbose

### set up observation and occlusion angles
opts.observation_angle *= utils.deg2rad

if opts.solar_occlusion_angle < 0.27125: ### occlusion angle is less than the radius of the sun
    print >> sys.stderr, "--solar-occlusion-angle is less than the radius of the Sun!. Increasing this to the radius of the Sun -> 0.27125 deg"
    opts.solar_occlusion_angle = 0.27125
opts.solar_occlusion_angle *= utils.deg2rad

if opts.lunar_occlusion_angle < 0.28416:
    print >> sys.stderr, "--lunar-occlusion-angle is less than the radius of the Moon! Increasing this to the radius of the Moon -> 0.28416 deg"
    opts.lunar_occlusion_angle = 0.28416
opts.lunar_occlusion_angle *= utils.deg2rad

### set up working NSIDE
NSIDE = 2**opts.nside_exp

### set up observatories and convert angles to radians
observatories = {}
observatories.update( 
    dict( (observatory, [utils.observatories[observatory].lat, utils.observatories[observatory].lon]) for observatory in opts.knownObservatory ) 
    )
observatories.update( 
    dict( (observatory, [float(lat)*utils.deg2rad, float(lon)*utils.deg2rad]) for observatory, lat, lon in opts.observatory ) 
    )
observatoryNames = sorted(observatories.keys())

### set seed so run is reproducible
if opts.seed!=None:
    np.random.seed(opts.seed)
if opts.Verbose:
    print "USING numpy.random state : %s"%(str(np.random.get_state()))

### set up tagging and output directory
if opts.tag:
    opts.tag = "_"+opts.tag
if opts.expensive_logging:
    opts.tag = "_eL"+opts.tag
if not os.path.exists(opts.output_dir):
    os.makedirs(opts.output_dir)

#-------------------------------------------------

### load library of FITS files
library = []
for index in catalogs:
    if opts.verbose:
        print "loading data from : %s"%(index)

    rootdir = os.path.dirname( index ) ### used to determine full paths to FITS files
    file_obj = open(index, "r")
    cols = dict( (c, i) for i, c in enumerate(file_obj.readline().strip().split()) ) ### get column headers

    if not cols.has_key(opts.algorithm): ### ensure we have LIB maps
        raise ValueError("could not find algorithm=\"%s\" as a column header in %s"%(opts.algorithm, index))
    if not cols.has_key("gps"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))
    if not cols.has_key("RA"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))
    if not cols.has_key("Dec"): ### ensure we have time for each map
        raise ValueError("could not find \"gps\" as a column header in %s"%(index))

    for line in file_obj:
        fields = line.strip().split()
        fits = fields[cols[opts.algorithm]]
        if fits!="none": ### there is a map for this injection
            fits = os.path.join(rootdir, fits)
            gps = float(fields[cols['gps']])
            ra = float(fields[cols['RA']])
            dec = float(fields[cols['Dec']])
            if opts.Verbose:
                print "    %.3f -> %s"%(gps, fits)
            library.append( {'gps':gps, 'fits':fits, 'ra':ra, 'dec':dec} )
    Nm = len(library)
    if opts.verbose:
        print "library now contains %d events"%(Nm)

#-------------------------------------------------

### iterate through trials
if opts.verbose:
    print "simulating %d trials"%opts.Nt

for i in xrange(opts.Nt):
    if opts.verbose:
        print "-------------\n    %d / %d\n-------------"%(i+1, opts.Nt)

    ### randomly draw events
    if opts.Verbose:
        print "finding %d times"%opts.Nd
    eventTimes = utils.drawTimes( start, end, opts.Nd, amplitude=opts.amplitude, phase=opts.phase, verbose=opts.Verbose )

    ### randomly draw skymaps from library and compute statistics
    skymaps = []
    for ind, (t, j) in enumerate(zip( eventTimes, np.random.randint( 0, high=Nm, size=opts.Nd ) )):
        d = library[j]
        if opts.verbose:
            print "    %d / %d : selected %s"%(ind+1, opts.Nd, d['fits'])
        if not d.has_key('post'): ### load map for the first time!
            if opts.Verbose:
                print "        loading skymap"
            d['post'] = hp.read_map( d['fits'], verbose=opts.Verbose )

        ### rotate maps to new coord systems based on "detection times"
        ### this is done because we expect the catalogs to accurately represent distributions of detected events
        ### and their source positions relative to the fixed detectors. Therefore, we must account for the 
        ### rotation of the detectors and map the position of the "detection" into new equatorial coordinates
        ### corresponding to the "detection time."

        ### note, we store the detection time, the map in Celestial coords and the map in EarthFixed coords
        ###    it may be faster to simply store the map in EarthFixed coords within the library when we load it...
        ###    this would simplify the rotation logic slightly...
        if opts.Verbose:
            print "        rotating map : %.9f -> %.9f"%(d['gps'], t)

        post = hp.ud_grade( utils.rotateMapC2C(d['post'], d['gps'], t), nside_out=NSIDE, power=-2)

        if opts.Verbose:
            print "        computing statistics"

        D = {}
        skymaps.append( D )
        ### get positions of the sun and moon
        sunDec, sunRA = utils.solarPosition( t, coord="C" )
        lunDec, lunRA = utils.lunarPosition( t, coord="C" )

        ### store basic parameters
        D.update( {'gps':t, 
                   'original_gps':d['gps'],
                   'dec':d['dec'], 'ra':utils.rotateRAC2C(d['ra'], d['gps'], t),
                   'sunDec':sunDec, 'sunRA':sunRA,
                   'lunDec':lunDec, 'lunRA':lunRA,
                   'fits':d['fits'],
                  }
                )

        ### compute occlusion masks 
        solarMask = utils.solarOcclusion( sunDec, sunRA, NSIDE, opts.solar_occlusion_angle, coord="C")
        lunarMask = utils.lunarOcclusion( lunDec, lunRA, NSIDE, opts.lunar_occlusion_angle, coord="C")
        mask = solarMask*lunarMask ### only keep things that are not occluded by either the sun or the moon
        maskedPost = post*mask

        if opts.expensive_logging: ### store expensive things...
            D.update( {'celestial':post,
                       'solarMask':solarMask,
                       'lunarMask':lunarMask,
                      }
                    )

        ### find source's true location
        dec = D['dec']
        ra = D['ra']
        srcTheta = utils.pi2-dec
        truePix = hp.ang2pix(NSIDE, srcTheta, ra) ### the location of the true source

        sum_maskedPost = np.sum(maskedPost)
        observable = mask[truePix]

        if opts.timeUntilUnOc:
            timeUntilUnOc= utils.timeUntilUnoccluded( dec, ra, sunDec, sunRA, lunDec, lunRA, t, opts.solar_occlusion_angle, opts.lunar_occlusion_angle )
        else:
            timeUntilUnOc = -1

        if opts.verbose:
            print "        observable conf = %.6e"%(sum_maskedPost)
            if observable:
                print "        source location IS observable"
            else:
                print "        source location IS NOT observable"
                if solarMask[truePix]:
                    print "            blocked by the Moon"
                elif lunarMask[truePix]:
                    print "            blocked by the Sun"
                else:
                    print "            blocked by the Sun and the Moon"
                if opts.timeUntilUnOc:
                    print "            have to wait %.3f days before source is unoccluded"%(timeUntilUnOc/86400.)

        D['sum_maskedPost'] = sum_maskedPost
        D['observable'] = observable
        D['timeUntilUnOc'] = timeUntilUnOc

        ### iterate through observatories and compute what fraction of each map is observable for that site
        ### we print the results to the terminal
        for observatory in observatoryNames:
            lat, lon  = observatories[observatory] ### where this observatory is located

            unrealObsReg = utils.unrealistic_observableRegion( NSIDE, opts.observation_angle, lat )
            observableRegion = utils.observableRegion( sunDec, sunRA, NSIDE, opts.observation_angle, lat, lon, solar_occlusion_angle=opts.solar_occlusion_angle )

            ### we compute the amount of the map that coulb be accessible by multiplying the posterior by the observable region
            ### we copmute the actual amount of confidence that is accessible by multiplying the posterior, the mask, and the observable region
            sum_obsPost = np.sum(post*unrealObsReg)
            sum_obsMaskedPost = np.sum(maskedPost*observableRegion)
            obsObservable = (mask*observableRegion)[truePix]

            timeUntilObs = utils.timeUntilObservable( lat, lon, dec, ra, sunDec, sunRA, t, opts.observation_angle, solarOcclusionAng=opts.solar_occlusion_angle )
            marge_timeUntilObs = utils.marge_timeUntilObservable( post, lat, lon, sunDec, sunRA, t, opts.observation_angle, solarOcclusionAng=opts.solar_occlusion_angle )

            zenithAng = utils.zenithAng( lat, dec, ra, sunDec, sunRA, t, opts.observation_angle, solarOcclusionAng=opts.solar_occlusion_angle )
            marge_zenithAng = utils.marge_zenithAng( post, lat, sunDec, sunRA, t, opts.observation_angle, solarOcclusionAng=opts.solar_occlusion_angle )

            if opts.verbose:
                print "      %s\n        possibly observable conf = %.6e\n        actually observable conf = %.6e"%(observatory, sum_obsPost, sum_obsMaskedPost)
                if obsObservable:
                    print "        source location IS observable by %s\n            have to wait %.3f hours"%(observatory, timeUntilObs/3600.)
                else:
                    print "        source location IS NOT observable by %s"%(observatory)
            D[observatory] = {'sum_obsPost':sum_obsPost, 'sum_obsMaskedPost':sum_obsMaskedPost, 'obsObservable':obsObservable, 'timeUntilObs':timeUntilObs, 'marge_timeUntilObs':marge_timeUntilObs, 'zenithAng':zenithAng, 'marge_zenithAng':marge_zenithAng}

            ### sanity checks!
            if obsObservable and (timeUntilObs>=np.infty): ### these can conflict because of the discretization of pixels (obsObservable) and actual angles (timeUntilObs)
                obsObservable = False
                if opts.Verbose:
                    print( '          source should be observable but timeUntilObs is infinite' )
                    print( '            lat = %.3f'%(lat*utils.rad2deg) )
                    print( '            dec = %.3f'%(d['dec']*utils.rad2deg) )

            ### clean up explicitly to help save on memory
            ### this should be cleaned up regardless of opts.memory_light becasue it is the end of the trial
            del unrealObsReg, observableRegion, sum_obsPost, sum_obsMaskedPost, obsObservable, timeUntilObs, marge_timeUntilObs, zenithAng, marge_zenithAng
        
        if opts.memory_light: ### remove this from the library element so we don't hold onto the memory for the whole library
            d.pop('post')
   
    ### write pickle object
    pklname = "%s/simDet%s-%d-%d.pkl"%(opts.output_dir, opts.tag, i, opts.Nd)
    if opts.verbose:
        print "saving : %s"%pklname
    file_obj = open(pklname, "w")
    pickle.dump( skymaps, file_obj )
    if opts.expensive_logging:
        pickle.dump( observatories, file_obj )
    file_obj.close()

    ### clean up explicitly to help save on memory
    ### this should be cleaned up regardless of opts.memory_light becasue it is the end of the trial
    del eventTimes, j, t, ind, skymaps, d, post, dec, ra, sunDec, sunRA, solarMask, lunDec, lunRA, lunarMask, mask, maskedPost, truePix, sum_maskedPost, observable, timeUntilUnOc, file_obj, srcTheta
