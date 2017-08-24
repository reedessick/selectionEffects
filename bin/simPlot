#!/usr/bin/python
usage = "simPlot.py [--options] simDet.pkl simDet.pkl simDet.pkl"
description  = """plot the results of simDet.py"""
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import os

import pickle

import numpy as np
import healpy as hp

import simUtils as utils
import simPlotUtils as putils

plt = putils.plt

from optparse import OptionParser

#-------------------------------------------------

parser = OptionParser(usage=usage, description=description)

### options about verbosity
parser.add_option("-v", "--verbose", default=False, action="store_true")
parser.add_option("-V", "--Verbose", default=False, action="store_true", help="print very detailed statements")

### mollweide plotting options
parser.add_option("", "--nside-exp", default=8, type="int", help="when stacking maps and comparing them, we resample the maps so that they are at nside=2**nside_exp")

parser.add_option("", "--indiv-Celestial", default=False, action="store_true", help="plot Mollweide projections in Celestial coordinates for all individual evnets")
parser.add_option("", "--indiv-EarthFixed", default=False, action="store_true", help="plot Mollweide projections in EarthFixed coordinates for all individual evnets")
parser.add_option("", "--indiv-contours", default=False, action="store_true", help="plot contours in \"--indiv-*\" Mollweide projections")

parser.add_option("", "--trial-Celestial", default=False, action="store_true", help="plot Mollweide projections in Celestial coordinates for all events belonging to each trial, but each trial separately.")
parser.add_option("", "--trial-EarthFixed", default=False, action="store_true", help="plot Mollweide projections in EarthFixed coordinates for all events belonging to each trial, but each trial separately.")
parser.add_option("", "--trial-contours", default=False, action="store_true", help="plot contours on \"--trial-*\" Mollweide projections")

parser.add_option("", "--all-Celestial", default=False, action="store_true", help="plot Mollweide projections in Celesital coordinates for all events from all trials together")
parser.add_option("", "--all-EarthFixed", default=False, action="store_true", help="plot Mollweide projections in EarthFixed coordinates for all events from all trials together")
parser.add_option("", "--all-contours", default=False, action="store_true", help="plot contours on \"--all-*\" Mollweide projections")

### options about occlusion
parser.add_option("-o", "--solar-occlusion-angle", default=30, type="float", help="the closest angle to the sun that is still observable. Should be specified in degrees so that the Sun will block out half the sky when --occlusion-angle=90.")
parser.add_option("-m", "--lunar-occlusion-angle", default=10, type="float", help="the closest angle to the sun that is still observable. Should be specified in degrees so that the Sun will block out half the sky when --occlusion-angle=90.")

### options about which observatories to use
parser.add_option("-S", "--observatory", dest="knownObservatory", default=[], action="append", type="string", help="a known observatory. Must be one of : 'Palomar','Cerro Tololo','Mauna Kea','La Palma','Anglo-Australian','Nishi-Harima'")

parser.add_option("-s", "--observatory-latitude-longitude", dest="observatory", nargs=3, default=[], action="append", type="string", help="the observatory name longitude and latitue as space separated strings, eg: Palomar XXX YYY. Latitude and Longitude should be specified in degrees.")

parser.add_option("-O", "--observation-angle", default=60, type="float", help="the maximum angle from the zenith which an observatory can observe. Should be specified in degrees.")

### parameters about saving the result
parser.add_option("-d", "--output-dir", default=".", type="string")
parser.add_option("-t", "--tag", default="", type="string")

parser.add_option("-f", "--figtype", default=[], action="append", type="string", help="the type of figure to save. We can save the same figure in multiple formats by repeating this option.")
parser.add_option("", "--saveFITS", default=False, action="store_true", help="save the FITS files associated with each figure")

opts, args = parser.parse_args()

if not args:
    raise ValueError("please supply at least one pickle file containing the output of simDet.py\n%s"%(description))

### verbosity statements
opts.verbose = opts.verbose or opts.Verbose

### set up tagging and output directory
if opts.tag:
    opts.tag = "_"+opts.tag
if not os.path.exists(opts.output_dir):
    os.makedirs(opts.output_dir)

### finish parsing --figtype
if not opts.figtype:
    opts.figtype.append( "png" )

### set up I/O flags
celestial = opts.indiv_Celestial or opts.trial_Celestial or opts.all_Celestial ### do we need maps in Celestial coords?
earthFixed = opts.indiv_EarthFixed or opts.trial_EarthFixed or opts.all_EarthFixed ### do we need maps in EarthFixed coords?

loadFITS = celestial or earthFixed ### do we need to load the FITS files?

timeLag = opts.trial_lag or opts.all_lag ### extract and format data for timeLag plotting

### check to see if there is anything to do
if not (celestial or earthFixed):
    if opts.verbose:
        print "please supply at least one of --indiv-Celestial, --indiv-EarthFixed, --trial-Celestial, --trial-EarthFixed, --all-Celestial, or --all-EarthFixed. None of these were supplied so there is nothing to do!"
    import sys
    sys.exit(0)

### set up parameters of working NSIDE
NSIDE = 2**opts.nside_exp
NPIX = hp.nside2npix(NSIDE)

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

### set up observatories and convert angles to radians
observatories = {}
observatories.update(
    dict( (observatory, [utils.observatories[observatory].lat, utils.observatories[observatory].lon]) for observatory in opts.knownObservatory )
    )
observatories.update(
    dict( (observatory, [float(lat)*utils.deg2rad, float(lon)*utils.deg2rad]) for observatory, lat, lon in opts.observatory )
    )
observatoryNames = sorted(observatories.keys())

#-------------------------------------------------

### load in data and plot (keeps processes ~light by plotting as we go)

if opts.all_Celestial: ### set up arrays for stacking events across all trials
    allCelestial        = np.zeros((NPIX,), dtype=float)
    allCelestial_masked = np.zeros((NPIX,), dtype=float)
    allCelestial_maskedObsReg = dict( (observatory, np.zeros((NPIX,), dtype=float)) for observatory in observatoryNames )
if opts.all_EarthFixed: ### set up arrays for stacking events across all trials
    allEarthFixed        = np.zeros((NPIX,), dtype=float)
    allEarthFixed_masked = np.zeros((NPIX,), dtype=float)
    allEarthFixed_maskedObsReg = dict( (observatory, np.zeros((NPIX,), dtype=float)) for observatory in observatoryNames )

for pkl in args: ### iterate through pkl files

    if opts.trial_Celestial: ### set up arrays for stacking events in this trial
        trialCelestial        = np.zeros((NPIX,), dtype=float)
        trialCelestial_masked = np.zeros((NPIX,), dtype=float)
        trialCelestial_maskedObsReg = dict( (observatory, np.zeros((NPIX,), dtype=float)) for observatory in observatoryNames )
    if opts.trial_EarthFixed: ### set up arrays for stacking events in this trial
        trialEarthFixed        = np.zeros((NPIX,), dtype=float)
        trialEarthFixed_masked = np.zeros((NPIX,), dtype=float)
        trialEarthFixed_maskedObsReg = dict( (observatory, np.zeros((NPIX,), dtype=float)) for observatory in observatoryNames )

    ### load the data from disk
    if opts.verbose:
        print "loading data from : %s"%pkl
    file_obj = open(pkl, "r")
    events = pickle.load(file_obj) ### do not assume expensive logging...
    file_obj.close()
    Nd = len(events)
    if opts.Verbose:
        print "    found %d events"%(Nd)

    #---------------------------------------------
    # plots that require FITS files
    #---------------------------------------------

    if loadFITS: ### we need to iterate through events and load FITS files
        figname_template = opts.output_dir+"/"+os.path.basename(pkl).strip(".pkl")+opts.tag+"-%s.%s"

        for i, d in enumerate(events):
            ### only assume d has the following keys
            ###   gps
            ###   original_gps
            ###   fits
            ###   dec
            ###   ra

            if opts.verbose:
                print "    -----\n    %d / %d\n    -----"%(i+1, Nd)

            if opts.Verbose:
                print "        loading : %s"%d['fits']
            post = hp.read_map( d['fits'], verbose=opts.Verbose )
            if len(post)!=NPIX: ### make sure we're working at the desired resolution 
                post = hp.ud_grade( post, nside_out=NSIDE, power=-2 ) 
   
            ### set up map and masks for Celestial plots
            if celestial:
                if opts.Verbose:
                    print "            rotating to new gps : %.6f -> %.6f"%(d['original_gps'], d['gps'])
                cpost = utils.rotateMapC2C(post, d['original_gps'], d['gps'])
                cmask = utils.solarOcclusion( d['sunDec'], d['sunRA'], NSIDE, opts.solar_occlusion_angle, coord="C")*utils.lunarOcclusion( d['lunDec'], d['lunRA'], NSIDE, opts.lunar_occlusion_angle, coord="C")
                cpost_masked = cpost*cmask

                observatoriesCelestial = {}
                for observatory in observatoryNames:
                    if opts.Verbose:
                        print "            computing observable region for %s"%(observatory)
                    ### compute observable region
                    lat, lon = observatories[observatory]
                    observatoriesCelestial[observatory] = utils.observableRegion( d['sunDec'], d['sunRA'], NSIDE, opts.observation_angle, lat, lon, solar_occlusion_angle=opts.solar_occlusion_angle )

            ### set up map and masks for EarthFixed plots
            if earthFixed:
                if opts.Verbose:
                    print "            rotating to EarthFixec coordinates"
                epost = utils.rotateMapC2E(post, d['original_gps'])
                emask = utils.solarOcclusion( d['sunDec'], d['sunRA'], NSIDE, opts.solar_occlusion_angle, coord="C")*utils.lunarOcclusion( d['lunDec'], d['lunRA'], NSIDE, opts.lunar_occlusion_angle, coord="C")
                emask = utils.rotateMapC2E( emask, d['gps'] )
                epost_masked = epost*emask

                observatoriesEarthFixed = {} 
                for observatory in observatoryNames:
                    if opts.Verbose:
                        print "            computing observable region for %s"%(observatory)
                    ### compute observable region
                    lat, lon = observatories[observatory]
                    observatoriesEarthFixed[observatory] = utils.rotateMapC2E( utils.observableRegion( d['sunDec'], d['sunRA'], NSIDE, opts.observation_angle, lat, lon, solar_occlusion_angle=opts.solar_occlusion_angle ), d['gps'] )

            del post ### we don't need this copy anymore...

            ### individual Mollweide Plotting for Celestial coordinates
            if opts.indiv_Celestial:
                if opts.Verbose:
                    print "        plotting Celestial projections"
                fig, axs = putils.mollweideScaffold( cpost, cpost_masked, mask=cmask, projection="astro mollweide", contours=opts.indiv_contours )
                for figtype in opts.figtype:
                    figname = figname_template%("celest-%d"%i, figtype)
                    if opts.verbose:
                        print "            "+figname
                        fig.savefig(figname)
                plt.close(fig)

                if opts.saveFITS: ### save into FITS files
                    fitsname = figname_template%("celest-%d"%i, 'fits.gz')
                    if opts.Verbose:
                        print "            "+fitsname
                    hp.write_map( fitsname, cpost )
                    fitsname = figname_template%("celest-%d_masked"%i, 'fits.gz')
                    if opts.verbose:
                        print "            "+fitsname
                    hp.write_map( fitsname, cpost_masked )

                ### observatories separately
                for observatory in observatoryNames:
                    if opts.Verbose:
                        print "        plotting Celestial projections for %s"%observatory

                    obsReg = observatoriesCelestial[observatory] 
                    fig, axs = putils.mollweideScaffold( cpost, cpost_masked*obsReg, mask=cmask*obsReg, projection="astro mollweide", contours=opts.indiv_contours )
                    for figtype in opts.figtype:
                        figname = figname_template%("celest_%s-%d"%(observatory,i), figtype)
                        if opts.verbose:
                            print "            "+figname
                        fig.savefig(figname)
                    plt.close(fig)

                    if opts.saveFITS: ### save into FITS files
                        fitsname = figname_template%("celest_%s-%d_masked"%(observatory,i), 'fits.gz')
                        if opts.verbose:
                            print "            "+fitsname
                        hp.write_map( fitsname, cpost_masked*obsReg )

            ### individual Mollweide PLotting for EarthFixed coordinates
            if opts.indiv_EarthFixed:
                if opts.Verbose:
                    print "        plotting EarthFixed projections"
                fig, axs = putils.mollweideScaffold( epost, epost_masked, mask=emask, projection="mollweide", contours=opts.indiv_contours )
                for figtype in opts.figtype:
                    figname = figname_template%("earth-%d"%i, figtype)
                    if opts.verbose:
                        print "            "+figname
                        fig.savefig(figname)
                plt.close(fig)

                if opts.saveFITS: ### save into FITS files
                    fitsname = figname_template%("earth-%d"%i, 'fits.gz')
                    if opts.Verbose:
                        print "            "+fitsname
                    hp.write_map( fitsname, epost )
                    fitsname = figname_template%("earth-%d_masked"%i, 'fits.gz')
                    if opts.Verbose:
                        print "            "+fitsname
                    hp.write_map( fitsname, epost_masked )

                ### observatories separately
                for observatory in observatoryNames:
                    if opts.Verbose:
                        print "        plotting EarthFixed projections for %s"%observatory
                   
                    obsReg = observatoriesEarthFixed[observatory] 
                    lat, lon = observatories[observatory]
                    fig, axs = putils.mollweideScaffold( epost, epost_masked*obsReg, mask=emask*obsReg, projection="mollweide", contours=opts.indiv_contours, LatLon=(lat,lon) )
                    for figtype in opts.figtype:
                        figname = figname_template%("earth_%s-%d"%(observatory,i), figtype)
                        if opts.verbose:
                            print "            "+figname
                        fig.savefig(figname)
                    plt.close(fig)
 
                    if opts.saveFITS: ### save into FITS files
                        fitsname = figname_template%("earth_%s-%d_masked"%(observatory,i), 'fits.gz')
                        if opts.Verbose:
                            print "            "+fitsname
                        hp.write_map( fitsname, epost_masked*obsReg )

            ### trial Mollweide Plotting
            if opts.trial_Celestial:
                if opts.Verbose:
                    print "        stacking trialCelestial"
                trialCelestial += cpost
                trialCelestial_masked += cpost_masked
                for observatory in observatoryNames:
                    obsReg = observatoriesCelestial[observatory] 
                    trialCelestial_maskedObsReg[observatory] += cpost_masked*obsReg

            if opts.trial_EarthFixed:
                if opts.Verbose:
                    print "        stacking trialEarthFixed"
                trialEarthFixed += epost
                trialEarthFixed_masked += epost_masked
                for observatory in observatoryNames:
                    obsReg = observatoriesEarthFixed[observatory]
                    trialEarthFixed_maskedObsReg[observatory] += epost_masked*obsReg

            ### all Mollweide Plotting
            if opts.all_Celestial:
                if opts.Verbose:
                    print "    stacking allCelestial"
                allCelestial += cpost
                allCelestial_masked += cpost_masked
                for observatory in observatoryNames:
                    obsReg = observatoriesCelestial[observatory] 
                    allCelestial_maskedObsReg[observatory] += cpost_masked*obsReg

            if opts.all_EarthFixed:
                if opts.Verbose:
                    print "    stacking allEarthFixed"
                allEarthFixed += epost
                allEarthFixed_masked += epost_masked
                for observatory in observatoryNames:
                    obsReg = observatoriesEarthFixed[observatory] 
                    allEarthFixed_maskedObsReg[observatory] += epost_masked*obsReg

    ### trial Mollweide Plotting
    if opts.trial_Celestial:
        if opts.Verbose:
            print "    plotting Celestial projections"
        fig, axs = putils.mollweideScaffold( trialCelestial, trialCelestial_masked, projection="astro mollweide", contours=opts.trial_contours )
        for figtype in opts.figtype:
            figname = figname_template%("celest", figtype)
            if opts.verbose:
                print "        "+figname
            fig.savefig(figname)
        plt.close(fig)

        if opts.saveFITS: ### save into FITS files
            fitsname = figname_template%("celest", 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, trialCelestial )
            fitsname = figname_template%("celest_masked", 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, trialCelestial_masked )

        for observatory in observatoryNames:
            if opts.Verbose:
                print "    plotting Celestial projections for %s"%observatory
            fig, axs = putils.mollweideScaffold( trialCelestial, trialCelestial_maskedObsReg[observatory], projection="astro mollweide", contours=opts.trial_contours )
            for figtype in opts.figtype:
                figname = figname_template%("celest_%s"%(observatory), figtype)
                if opts.verbose:
                    print "        "+figname
                fig.savefig(figname)
            plt.close(fig)

            if opts.saveFITS: ### save into FITS files
                fitsname = figname_template%("celest_%s_masked"%(observatory), 'fits.gz')
                if opts.Verbose:
                    print "            "+fitsname
                hp.write_map( fitsname, trialCelestial_maskedObsReg[observatory] )


    if opts.trial_EarthFixed:
        if opts.Verbose:
            print "    plotting EarthFixed projections"
        fig, axs = putils.mollweideScaffold( trialEarthFixed, trialEarthFixed_masked, projection="mollweide", contours=opts.trial_contours )
        for figtype in opts.figtype:
            figname = figname_template%("earth", figtype)
            if opts.verbose:
                print "        "+figname
            fig.savefig(figname)
        plt.close(fig)

        if opts.saveFITS: ### save into FITS files
            fitsname = figname_template%("earth", 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, trialEarthFixed )
            fitsname = figname_template%("earth", 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, trialEarthFixed_masked )

        for observatory in observatoryNames:
            if opts.Verbose:
                print "    plotting EarthFixed projections for %s"%observatory
            lat, lon = observatories[observatory]
            fig, axs = putils.mollweideScaffold( trialEarthFixed, trialEarthFixed_maskedObsReg[observatory], projection="mollweide", contours=opts.indiv_contours, LatLon=(lat,lon) )
            for figtype in opts.figtype:
                figname = figname_template%("earth_%s"%(observatory), figtype)
                if opts.verbose:
                    print "        "+figname
                fig.savefig(figname)
            plt.close(fig)

            if opts.saveFITS: ### save into FITS files
                fitsname = figname_template%("earth_%s_masked"%(observatory), 'fits.gz')
                if opts.Verbose:
                    print "            "+fitsname
                hp.write_map( fitsname, trialEarthFixed_maskedObsReg[observatory] )

#-------------------------------------------------
# plots that require FITS files
#-------------------------------------------------

figname_template = opts.output_dir+"/simPlot"+opts.tag+"-%s.%s"

### plot all events together
if opts.all_Celestial:
    if opts.Verbose:
        print "plotting Celestial projections"
    fig, axs = putils.mollweideScaffold( allCelestial, allCelestial_masked, projection="astro mollweide", contours=opts.all_contours )
    for figtype in opts.figtype:
        figname = figname_template%("celest", figtype)
        if opts.verbose:
            print "    "+figname
        fig.savefig(figname)
    plt.close(fig)

    if opts.saveFITS: ### save into FITS files
        fitsname = figname_template%("celest", 'fits.gz')
        if opts.Verbose:
            print "            "+fitsname
        hp.write_map( fitsname, allCelestial )
        fitsname = figname_template%("celest_masked", 'fits.gz')
        if opts.Verbose:
            print "            "+fitsname
        hp.write_map( fitsname, allCelestial_masked )

    for observatory in observatoryNames:
        if opts.Verbose:
            print "plotting Celestial projections for %s"%observatory
        fig, axs = putils.mollweideScaffold( allCelestial, allCelestial_maskedObsReg[observatory], projection="astro mollweide", contours=opts.all_contours )
        for figtype in opts.figtype:
            figname = figname_template%("celest_%s"%(observatory), figtype)
            if opts.verbose:
                print "    "+figname
            fig.savefig(figname)
        plt.close(fig)

        if opts.saveFITS: ### save into FITS files
            fitsname = figname_template%("celest_%s_masked"%(observatory), 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, allCelestial_maskedObsReg[observatory] )


if opts.all_EarthFixed:
    if opts.Verbose:
        print "plotting EarthFixed projections"
    fig, axs = putils.mollweideScaffold( allEarthFixed, allEarthFixed_masked, projection="mollweide", contours=opts.all_contours )
    for figtype in opts.figtype:
        figname = figname_template%("earth", figtype)
        if opts.verbose:
            print "    "+figname
        fig.savefig(figname)
    plt.close(fig)

    if opts.saveFITS: ### save into FITS files
        fitsname = figname_template%("earth", 'fits.gz')
        if opts.Verbose:
            print "            "+fitsname
        hp.write_map( fitsname, allEarthFixed )
        fitsname = figname_template%("earth_masked", 'fits.gz')
        if opts.Verbose:
            print "            "+fitsname
        hp.write_map( fitsname, allEarthFixed_masked )

    for observatory in observatoryNames:
        if opts.Verbose:
            print "plotting EarthFixed projections for %s"%observatory
        lat, lon = observatories[observatory][:2]
        fig, axs = putils.mollweideScaffold( allEarthFixed, allEarthFixed_maskedObsReg[observatory], projection="mollweide", contours=opts.all_contours, LatLon=(lat,lon) )
        for figtype in opts.figtype:
            figname = figname_template%("earth_%s"%(observatory), figtype)
            if opts.verbose:
                print "    "+figname
            fig.savefig(figname)
        plt.close(fig)

        if opts.saveFITS: ### save into FITS files
            fitsname = figname_template%("earth_%s_masked"%(observatory), 'fits.gz')
            if opts.Verbose:
                print "            "+fitsname
            hp.write_map( fitsname, allEarthFixed_maskedObsReg[observatory] )
