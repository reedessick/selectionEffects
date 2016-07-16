#!/usr/bin/python
usage = "simSum.py [--options] pkl pkl pkl..."
description = "generate summary information about a set of simulations. Currently, this means making big corner plots..."
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import pickle

import numpy as np

import simUtils as utils

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import corner

from optparse import OptionParser

#-------------------------------------------------

parser = OptionParser( usage=usage, description=description )

parser.add_option("-v", "--verbose", default=False, action="store_true")

### compute for which observatories
parser.add_option("-s", "--observatory", dest="observatory", nargs=1, default=[], action="append", type="string", help="the observatory name, eg: Palomar. NOTE: the script only generates observatory-specific information for explicitly specified sites and assumes all pkl files contain data for all explicitly specified sites.")

### output format
parser.add_option('-d', '--output-dir', default='.', type='string')
parser.add_option('-t', '--tag', default='', type='string')
parser.add_option('', '--figtype', default=[], action='append', type='string', help='default="png"')

parser.add_option('', '--skip-N', default=False, action="store_true", help="do not plot the number of detections in corner plot")
parser.add_option('', '--skip-Ds', default=False, action="store_true", help="do not plot the Dzenith or Ddelay in corner plots")

parser.add_option('', '--exclude-unphysical', default=False, action="store_true", help="set range values to exclude unphysical numbers. Relevant for Ds")

opts, args = parser.parse_args()

if opts.tag:
    opts.tag = "_"+opts.tag
if not opts.figtype:
    opts.figtype.append( "png" )

#-------------------------------------------------

### read in data
if opts.verbose:
    print "loading data"

N                    = []
observable           = []
obsMaskedProb        = []
observableSite       = dict( (observatory, []) for observatory in opts.observatory )
obsProbSite          = dict( (observatory, []) for observatory in opts.observatory )
obsMaskedProbSite    = dict( (observatory, []) for observatory in opts.observatory )
timeTilObsSite       = dict( (observatory, []) for observatory in opts.observatory )
marge_timeTilObsSite = dict( (observatory, []) for observatory in opts.observatory )
zenithAng            = dict( (observatory, []) for observatory in opts.observatory )
marge_zenithAng      = dict( (observatory, []) for observatory in opts.observatory )
for pkl in args:
    if opts.verbose:
        print "    reading data from : %s"%(pkl)
    file_obj = open( pkl, "r" )
    data = pickle.load( file_obj )
    file_obj.close()

    if opts.verbose:
        print "        extracting statistics" 
    ### extract data
    N.append( len(data) )

    ### we take the mean over all detections in the trial. We can investigate the dependence on Nd by running trials with different numbers of detections
    observable.append( np.mean( [d['observable'] for d in data] ) )
    obsMaskedProb.append( np.mean( [d['sum_maskedPost'] for d in data] ) )

    ### we take the mean of statistics for each observatory separately as well
    for observatory in opts.observatory:
        observableSite[observatory].append( np.mean( [d[observatory]['obsObservable'] for d in data] ) )
        obsProbSite[observatory].append( np.mean( [d[observatory]['sum_obsPost'] for d in data] ) )

        pobs = [d[observatory]['sum_obsMaskedPost'] for d in data]
        obsMaskedProbSite[observatory].append( np.mean( pobs ) )
        pobs = np.sum( pobs )

        ### add zenith angle
        if not opts.skip_Ds:
            zA  = []
            mzA = 0.0
            tD  = []
            mtD = 0.0
            for d in data:
                if d[observatory]['obsObservable']:
                    if (d[observatory]['timeUntilObs']<np.infty):
                        tD.append( d[observatory]['timeUntilObs'] )

                    zA.append( d[observatory]['zenithAng']*utils.rad2deg )

                if d[observatory]['sum_obsMaskedPost'] > 0:
                    mtD += d[observatory]['marge_timeUntilObs'].real

                    mzA += d[observatory]['marge_zenithAng'].real*utils.rad2deg 

            if zA:
                zenithAng[observatory].append( np.mean( zA ) )
            else:
                zenithAng[observatory].append( -45 ) ### unphysical

            if mzA:
                marge_zenithAng[observatory].append( mzA / pobs )
            else:
                marge_zenithAng[observatory].append( -45 ) ### unphysical

            if tD:
                timeTilObsSite[observatory].append( np.mean(tD) )
            else:
                timeTilObsSite[observatory].append( -12*3600 ) ### push to a negative number so that it is unphysical

            if mtD:
                marge_timeTilObsSite[observatory].append( mtD / pobs )
            else:
                marge_timeTilObsSite[observatory].append( -12*3600 ) ### unphysical

### format data nicely
if opts.skip_N:
    data = np.transpose(np.array([observable, obsMaskedProb]))
    names = ['observable', 'observable\nprobability']
    dims = [(0,1), (0,1)]

    if opts.skip_Ds:
        siteData = dict( (observatory, np.transpose(np.array([observable, obsMaskedProb, observableSite[observatory], obsProbSite[observatory], obsMaskedProbSite[observatory]]))) for observatory in opts.observatory )
        siteNames = dict( (observatory, ['observable', 'observable\nprobability', 'observable from\n%s'%observatory, 'possibly observable\nprobability from\n%s'%observatory, 'actually observable\nprobability from\n%s'%observatory]) for observatory in opts.observatory )
        siteDims = [(0,1), (0,1), (0,1), (0,1), (0,1)]
    else:
        siteData = dict( (observatory, np.transpose(np.array([observable, obsMaskedProb, observableSite[observatory], obsProbSite[observatory], obsMaskedProbSite[observatory], timeTilObsSite[observatory], marge_timeTilObsSite[observatory], zenithAng[observatory], marge_zenithAng[observatory] ]))) for observatory in opts.observatory )
        siteNames = dict( (observatory, ['observable', 'observable\nprobability', 'observable from %s'%observatory, 'possibly observable\nprobability from\n%s'%observatory, 'actually observable\nprobability from\n%s'%observatory, '$\hat{D}^{(\mathrm{src})}_\mathrm{delay}$ from\n%s'%observatory, '$\hat{D}_\mathrm{delay}$ from\n%s'%observatory, '$\hat{D}^{(\mathrm{src})}_\mathrm{zenith}$ from\n%s'%observatory, '$\hat{D}_\mathrm{zenith}$ from\n%s'%observatory]) for observatory in opts.observatory )
        if opts.exclude_unphysical:
            siteDims = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,86400), (0,86400), (0, 180), (0,180)]
        else:
            siteDims = [(0,1), (0,1), (0,1), (0,1), (0,1), (-43200,86400), (-43200,86400), (-45, 180), (-45,180)]

else:
    data = np.transpose(np.array([N, observable, obsMaskedProb]))
    names = ['N', 'observable', 'observable\nprobability']
    dims = [(np.min(N)-1,np.max(N)+1), (0,1), (0,1)]

    if opts.skip_Ds:
        siteData = dict( (observatory, np.transpose(np.array([N, observable, obsMaskedProb, observableSite[observatory], obsProbSite[observatory], obsMaskedProbSite[observatory]]))) for observatory in opts.observatory )
        siteNames = dict( (observatory, ['N', 'observable', 'observable\nprobability', 'observable from\n%s'%observatory, 'possibly observable\nprobability from\n%s'%observatory, 'actually observable\nprobability from\n%s'%observatory]) for observatory in opts.observatory )
        siteDims = [(np.min(N)-1,np.max(N)+1), (0,1), (0,1), (0,1), (0,1), (0,1)]
    else:
        siteData = dict( (observatory, np.transpose(np.array([N, observable, obsMaskedProb, observableSite[observatory], obsProbSite[observatory], obsMaskedProbSite[observatory], timeTilObsSite[observatory], marge_timeTilObsSite[observatory], zenithAng[observatory], marge_zenithAng[observatory] ]))) for observatory in opts.observatory )
        siteNames = dict( (observatory, ['N', 'observable', 'observable\nprobability', 'observable from\n%s'%observatory, 'possibly observable\nprobability from\n%s'%observatory, 'actually observable\nprobability from\n%s'%observatory, '$\hat{D}^{(\mathrm{src})}_\mathrm{delay}$ from\n%s'%observatory, '$\hat{D}_\mathrm{delay}$ from\n%s'%observatory, '$\hat{D}^{(\mathrm{src})}_\mathrm{zenith}$ from\n%s'%observatory, '$\hat{D}_\mathrm{zenith}$ from\n%s'%observatory]) for observatory in opts.observatory )
        if opts.exclude_unphysical:
            siteDims = [(np.min(N)-1,np.max(N)+1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,86400), (0,86400), (0, 180), (0, 180)]
        else:
            siteDims = [(np.min(N)-1,np.max(N)+1), (0,1), (0,1), (0,1), (0,1), (0,1), (-43200,86400), (-43200,86400), (-45, 180), (-45, 180)]

### clean up some stuff to prevent multiple copies of the same thing from lingering
del file_obj, N, observable, obsMaskedProb, observableSite, obsProbSite, obsMaskedProbSite, timeTilObsSite

#------------------------

### plot/report/represent data

### marginalized over latitude
fig = corner.corner( data, labels=names, show_titles=True, range=dims )

fig.text(0.75, 0.75, '%d trials'%(len(args)), ha='center', va='center')

for figtype in opts.figtype:
    figname = "%s/simSum%s.%s"%(opts.output_dir, opts.tag, figtype)
    if opts.verbose:
        print "saving : %s"%(figname)
    fig.savefig( figname )
plt.close(fig)

### for each site separately
for observatory in opts.observatory:
    fig = corner.corner( siteData[observatory], labels=siteNames[observatory], show_titles=True, range=siteDims )

    fig.text(0.75, 0.75, '%d trials'%(len(args)), ha='center', va='center')

    for figtype in opts.figtype:
        figname = "%s/simSum%s-%s.%s"%(opts.output_dir, opts.tag, observatory, figtype)
        if opts.verbose:
            print "saving : %s"%(figname)
        fig.savefig( figname )
    plt.close(fig)
